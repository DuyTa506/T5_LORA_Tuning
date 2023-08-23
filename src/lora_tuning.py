import os
from typing import Generator
from typing import Iterator

import numpy as np
from accelerate import Accelerator # distributed training with multi-gpus
import torch
from utils import load_config
from datasets import load_dataset
from sophia import SophiaG
from dataset import MonoDataset
from dataset import ParallelDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import T5ForConditionalGeneration
from transformers import T5Tokenizer
from utils import calculate_bleu_score
from utils import create_logger
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from transformers import get_scheduler, set_seed, DataCollatorForSeq2Seq
EXPERIMENT_PATH = "experiments"
EXPERIMENT_CONFIG_NAME = "t5_config.yml"

os.environ["CUDA_VISIBLE_DEVICES"] = "0" # need single GPU or parallel GPU with comparision cache



class Lora_Trainer :
    """
    This is a trainer class for finetuning Huggingface T5 implementation with Lora tuning on custom dataset.

    Attributes:
        model_config: model configuration
        train_config: training configuration
        data_config: data paths configuration
        lora_config : lora adpater configuration
        experiment_path: path where experiment output will be dumped
        tokenizer: tokenizer
        device: device where experiment will happen (gpu or cpu)
        logger: File and terminal logger
    """
    def __init__(self,config : dict, experiment_path : str):
        self.data_config = config['data']
        self.model_config = config['model']
        self.lora_config = config['lora']
        self.train_config = config["training"]
        self.data_HF_config = config['data_config']
        self.experiment_path = experiment_path
        self.logger = create_logger(os.path.join(self.experiment_path, "log.txt"))
        if self.model_config["tokenizer_path"]:
            self.tokenizer = T5Tokenizer.from_pretrained(self.model_config["tokenizer_path"])
        else:
            self.logger.warning(
                f"No tokenizer path inputed, using {self.model_config['model_size']} default pretrained tokenizer")
            self.tokenizer = T5Tokenizer.from_pretrained(self.model_config["model_size"])
        self.device = torch.device(
            "cuda" if self.train_config["use_cuda"] and torch.cuda.is_available() else "cpu"
        )
        self.logger.info(f"Experiment Output Path: \n {self.experiment_path}")
        self.logger.info(f"Training will be begin with this configuration: \n {config} ")
    
    def _create_datasets(self) -> tuple:

        """
        Creates the following dataset from the data paths in the config file.

        - a train generator that generates batches of src and tgt data
        - a dev generator that generates batches of src dev data
        - tgt_dev that denotes the raw target dev data
        """
        # add task prefix and EOS token as required by model
        src_train = [
            self.data_config["src_prefix"]+":" + text.strip() + " </s>"
            for text in list(open(self.data_config["src_train"], encoding="utf-8"))
        ]
        src_dev = [
             self.data_config["src_prefix"] +":"+ text.strip() + " </s>"
            for text in list(open(self.data_config["src_dev"], encoding="utf-8"))
        ]

        tgt_train = [text.strip() + " </s>" for text in list(open(self.data_config["tgt_train"], encoding="utf-8"))]
        tgt_dev = [text.strip() for text in list(open(self.data_config["tgt_dev"], encoding="utf-8"))]

        # tokenize src and target data
        src_train_dict = self.tokenizer.batch_encode_plus(
            src_train,
            max_length=self.train_config["max_output_length"],
            return_tensors="pt",
            pad_to_max_length=True,
        )
        src_dev_dict = self.tokenizer.batch_encode_plus(
            src_dev,
            max_length=self.train_config["max_output_length"],
            return_tensors="pt",
            pad_to_max_length=True,
        )
        tgt_train_dict = self.tokenizer.batch_encode_plus(
            tgt_train,
            max_length=self.train_config["max_output_length"],
            return_tensors="pt",
            pad_to_max_length=True,
        )

        # obtain input tensors
        input_ids = src_train_dict["input_ids"]
        input_dev_ids = src_dev_dict["input_ids"]
        output_ids = tgt_train_dict["input_ids"]

        # specify data loader params and create train generator
        params = {
            "batch_size": self.train_config["batch_size"],
            "shuffle": self.train_config["shuffle_data"],
            "num_workers": self.train_config["num_workers_data_gen"],
        }
        train_generator = DataLoader(ParallelDataset(input_ids, output_ids), **params)
        print("train_generator:", train_generator)
        self.logger.info(f"Created training dataset of {len(input_ids)} parallel sentences")

        dev_params = params
        dev_params["shuffle"] = False
        dev_generator = DataLoader(MonoDataset(input_dev_ids), **dev_params)

        all_data = (train_generator, dev_generator, tgt_dev)

        return all_data


    def _create_hf_datasets(self):
        # Load Opus-100 English-Vietnamese dataset from Hugging Face datasets library
        dataset = load_dataset(self.data_HF_config["hf_dataset_name"], self.data_HF_config["hf_dataset_config"])

        train_dataset = dataset['train']
        test_dataset = dataset['test']
        
        LANG_TOKEN_MAPPING = {
            self.data_HF_config['src_lang']: '',
            self.data_HF_config['tgt_lang']: ''
        }
        
        special_tokens_dict = {'additional_special_tokens': list(LANG_TOKEN_MAPPING.values())}
        self.tokenizer.add_special_tokens(special_tokens_dict)
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        def encode_input_str(text, target_lang, tokenizer, seq_len,
                            lang_token_map=LANG_TOKEN_MAPPING):
            target_lang_token = lang_token_map[target_lang]

        # Tokenize and add special tokens
            input_ids = tokenizer.encode(
                text = target_lang_token + text,
                return_tensors = 'pt',
                padding = 'max_length',
                truncation = True,
                max_length = seq_len)

            return input_ids[0]
        
        def encode_target_str(text, tokenizer, seq_len,
                            lang_token_map=LANG_TOKEN_MAPPING):
            token_ids = tokenizer.encode(
                text = text,
                return_tensors = 'pt',
                padding = 'max_length',
                truncation = True,
                max_length = seq_len)
            
            return token_ids[0]
        def format_translation_data(translations, lang_token_map,
                                tokenizer, seq_len):
    # Choose a random 2 languages for in i/o
            langs = list(lang_token_map.keys())
            input_lang, target_lang = np.random.choice(langs, size=2, replace=False)

            # Get the translations for the batch
            input_text = translations[input_lang]
            target_text = translations[target_lang]

            if input_text is None or target_text is None:
                return None

            input_token_ids = encode_input_str(
                input_text, target_lang, tokenizer, seq_len, lang_token_map)
            
            target_token_ids = encode_target_str(
                target_text, tokenizer, seq_len, lang_token_map)

            return input_token_ids, target_token_ids

        def transform_batch(batch, lang_token_map, tokenizer):
            inputs = []
            targets = []
            for translation_set in batch['translation']:
                formatted_data = format_translation_data(
                    translation_set, lang_token_map, tokenizer, self.train_config['max_output_length'])
                
                if formatted_data is None:
                    continue

                input_ids, target_ids = formatted_data
                inputs.append(input_ids.unsqueeze(0))
                targets.append(target_ids.unsqueeze(0))
                
            batch_input_ids = torch.cat(inputs).cuda()
            batch_target_ids = torch.cat(targets).cuda()

            return batch_input_ids, batch_target_ids

        def get_data_generator(dataset, lang_token_map, tokenizer, batch_size=self.train_config['batch_size']):
            dataset = dataset.shuffle()
            for i in range(0, len(dataset), batch_size):
                raw_batch = dataset[i:i+batch_size]
                yield transform_batch(raw_batch, lang_token_map, tokenizer)



        target_labels = [data[self.data_HF_config['tgt_lang']] for data in test_dataset["translation"]]
        train_generator = get_data_generator(train_dataset, LANG_TOKEN_MAPPING, self.tokenizer)
        dev_generator = get_data_generator(test_dataset, LANG_TOKEN_MAPPING, self.tokenizer)

        all_data = (train_generator, dev_generator, target_labels)

        return all_data

    def _build_model(self) -> None:
        """
        Build model and update its configuration.
        """
        model_config = {
          "translation_en_to_vi": {
            "early_stopping": self.train_config["early_stopping"],
            "max_length": self.train_config["max_output_length"],
            "num_beams": self.train_config["beam_size"],
            "prefix": self.data_config["src_prefix"],
            
            
            
        }}
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_config["model_size"],device_map = "auto", load_in_8bit=self.train_config['use_int8'])
        self.model.config.update(model_config)
        ######## Lora adaptation ##########
        if self.lora_config['is_activate'] == True:
            if self.lora_config['checkpoint'] :
                accelerator.print(
                f"Loading pretrained peft model from {config['lora']['checkpoint']}"
            )
                self.model = PeftModel.from_pretrained(
                self.model, config["lora"]["checkpoint"], is_trainable=True
            )
            else:
                self.peft_config = LoraConfig(
                    r=self.lora_config["r"],
                    inference_mode=False,
                    lora_alpha=self.lora_config["alpha"],
                    lora_dropout=self.lora_config["lora_dropout"],
                    bias=self.lora_config['bias'],
                    task_type=TaskType.SEQ_2_SEQ_LM,
                    target_modules = self.lora_config["target_modules"]
                )
                self.model = get_peft_model(self.model, self.peft_config)
                self.model.print_trainable_parameters()
        
    def _build_optimizer(self, model_parameters: Iterator) -> torch.optim.Optimizer:
        """
        Build optimizer to be used in training.
        """
        if self.train_config["optimizer"] == "adam":
            optimizer = torch.optim.Adam(
                model_parameters,
                weight_decay=self.train_config["weight_decay"],
                lr=self.train_config["learning_rate"],
            )
        elif self.train_config["optimizer"] == "sgd":
            optimizer = torch.optim.SGD(
                model_parameters,
                weight_decay=self.train_config["weight_decay"],
                lr=self.train_config["learning_rate"],
            )
        elif self.train_config['optimizer'] == 'sophia':
            optimizer_cls = SophiaG
            optimizer = optimizer_cls(
            model_parameters,
            lr=self.train_config["learning_rate"],
            weight_decay=self.train_config["weight_decay"],
    )
        else:
            self.logger.warning(
                "Only 'adam' and 'sgd' and sophia is currently supported. Will use adam as default"
            )
            optimizer = torch.optim.Adam(
                model_parameters,
                weight_decay=self.train_config["weight_decay"],
                lr=self.train_config["learning_rate"],
            )
        return optimizer
    def train(self) -> None:
        """
        Train model on parallel dataset, evaluate dev data and save best model according to bleu if
        specified.
        """
        self.logger.info("Building model...")
        self._build_model()
        self.model.to(self.device)
        self.logger.info(f"Training will be done with this configuration: \n {self.model.config}")
        optimizer = self._build_optimizer(self.model.parameters())
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            "min",
            self.train_config["reduction_factor"],
            self.train_config["patience"],
            verbose=True,
            min_lr=self.train_config["min_lr"],
        )

        self.logger.info("Building data generators...")
        if self.data_HF_config['use_HF']:
            print('===================================')
            print('Preparing for huggingface dataset: ')

            train_generator, dev_generator, tgt_dev = self._create_hf_datasets()
            print("Done !!!")
            print('===================================')
            
        else :

            train_generator, dev_generator, tgt_dev = self._create_datasets()

        self.logger.info("Starting training...")
        best_bleu = 0.0
        for epoch in range(self.train_config["epochs"]):
            total_epoch_loss = 0.0
            running_loss = 0.0
            counter = 0

            for src_batch, tgt_batch in tqdm(train_generator):

                src_batch, tgt_batch = src_batch.to(self.device), tgt_batch.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(input_ids=src_batch, labels=tgt_batch)
                loss = outputs[0]
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                total_epoch_loss += loss.item()

                # print loss every 400 steps
                if counter % 400 == 0:
                    tqdm.write(
                        "Epoch: %d, Step: %5d, loss: %.3f"
                        % (epoch + 1, counter + 1, running_loss / 400)
                    )
                    running_loss = 0.0
                counter += 1
            self.logger.info(
                f"Epoch {epoch+1} done. Average Epoch Loss: {total_epoch_loss/counter}"
            )

            # evaluate dev and save model if new best bleu score
            if self.train_config["evaluate_dev"]:
                self.logger.info("Evaluating dev set...")
                bleu_score = self._evaluate_dev(dev_generator, tgt_dev, epoch)
                if bleu_score > best_bleu:
                    self._save_model(optimizer, epoch, loss.item(), best_bleu, self.model)
                    best_bleu = bleu_score
                else:
                    self.logger.info(
                        f"New BLEU not better than best BLEU - {best_bleu:.2f}, model not saved"
                    )
                self.model.train()

            if self.train_config["reduce_lr_on_bleu_plateau"]:
                scheduler.step(bleu_score)

        self.logger.info(
            f"Training done! All outputs saved in {self.experiment_path}. Best BLEU score was {best_bleu}"
        )

    def _save_model(
        self, optimizer: torch.optim.Optimizer, epoch: int, loss: float, best_bleu: float, model : torch.nn.Module
    ) -> None:
        """
        Save model.
        """
        state = {
            "epoch": epoch,
            "best_bleu_score": best_bleu,
            "loss": loss,
            "model_state": self.model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        }
        model_path = os.path.join(self.experiment_path, "best_model.pt")
        #adapter save
        torch.save(state,model_path)
        peft_model_id = f"{self.experiment_path}/{self.model_config['save_name']}_{self.peft_config.peft_type}_{self.peft_config.task_type}"
        model.save_pretrained(peft_model_id)
        self.logger.info(f"New best bleu! Model saved to {model_path}")
    
    def _evaluate_dev(
        self, dev_generator: Generator[torch.Tensor, None, None], tgt_dev: list, epoch: int,
    ) -> float:
        """
        Evaluate parallel dev dataset, write hypothesis to file, and displays bleu score.
        """

        # evaluate parallel dev dataset
        hyps = []
        self.model.eval()
        with torch.no_grad():
            for src_batch in tqdm(dev_generator):
                batch = src_batch.to(self.device)
                translations = self.model.generate(input_ids=batch)
                decoded = [
                    self.tokenizer.decode(
                        translation, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )
                    for translation in translations
                ]
                hyps = hyps + decoded
        bleu_score = calculate_bleu_score(hyps, tgt_dev)

        # write hypothesis to file
        hyps_path = os.path.join(self.experiment_path, f"epoch_{epoch+1}_dev_hyps.txt")
        with open(hyps_path, "w") as file:
            for sent in hyps:
                file.write(sent + " \n")
        self.logger.info(f"Model hypothesis saved in {hyps_path}")
        self.logger.info(f"BLEU score after epoch {epoch+1} is: {bleu_score:.2f}")

        return bleu_score
 
