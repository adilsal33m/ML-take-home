import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from zipfile import ZipFile
from tqdm import tqdm
from datasets import load_dataset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score
from transformers import ViTImageProcessor, DefaultDataCollator, AutoModelForImageClassification, TrainingArguments, Trainer, AutoImageProcessor, AutoModelForImageClassification, pipeline
from torchvision.transforms import RandomResizedCrop, RandomRotation, Compose, Normalize, ToTensor

# Constants
ROOT = r"E:/dev/ML-Take-Home/"
DATASET_PATH = f"{ROOT}dataset/potato_plans_diseases.zip"
EXTRACTED_IMAGES = f"{ROOT}dataset/images/"


def get_preprocessors():
    # Load image processor
    feature_extractor = ViTImageProcessor.from_pretrained(
        "google/vit-base-patch16-224-in21k")

    # Compose transformation for image: Rotate, ResizedCrop, Normalize
    normalize = Normalize(mean=feature_extractor.image_mean,
                          std=feature_extractor.image_std)
    _transforms = Compose([RandomRotation(degrees=(0, 180)), RandomResizedCrop(
        size=(224, 224)), ToTensor(), normalize])

    def transforms(examples):
        examples["pixel_values"] = [_transforms(
            img.convert("RGB")) for img in examples["image"]]
        del examples["image"]
        return examples

    return (feature_extractor, transforms)

def calculate_learning_rate(val):
    if val == 1:
        return 5e-5
    elif val == 2:
        return 1e-5
    elif val == 3:
        return 5e-6
    else:
        return 1e-6


def calculate_batch_size(val):
    if val == 1:
        return 16
    elif val == 2:
        return 8
    else:
        return 4


if __name__ == "__main__":
    with tqdm(total=100) as pbar:
        ##############################################################
        # Load dataset
        ##############################################################
        if os.path.exists(EXTRACTED_IMAGES):
            shutil.rmtree(EXTRACTED_IMAGES)

        pbar.update(10)
        pbar.set_description("Loading Dataset")

        # Extracting zip
        os.chdir(ROOT)
        with ZipFile(DATASET_PATH, 'r') as f:
            f.extractall(path=EXTRACTED_IMAGES)

        # Renaming folder for compatibility with datasets library
        os.rename(f'{EXTRACTED_IMAGES}PLD_3_Classes_256/Training',
                  f'{EXTRACTED_IMAGES}PLD_3_Classes_256/train')
        os.rename(f'{EXTRACTED_IMAGES}PLD_3_Classes_256/Testing',
                  f'{EXTRACTED_IMAGES}PLD_3_Classes_256/test')
        os.rename(f'{EXTRACTED_IMAGES}PLD_3_Classes_256/Validation',
                  f'{EXTRACTED_IMAGES}PLD_3_Classes_256/validation')

        # Load dataset through datasets library
        dataset = load_dataset(f"{EXTRACTED_IMAGES}PLD_3_Classes_256")

        # FOR TEST_ONLY
        # dataset = dataset['train'].train_test_split(test_size=0.99)
        # dataset['validation'] = dataset['test'].train_test_split(
        #     test_size=0.99)['train']
        # dataset['test'] = dataset['test'].train_test_split(
        #     test_size=0.9)['train']
        # TEST_ONLY

        pbar.update(20)
        pbar.set_description("Creating label dictionaries")

        # Create labels and label2id/id2label dicts
        labels = dataset["train"].features["label"].names
        label2id, id2label = dict(), dict()
        for i, label in enumerate(labels):
            label2id[label] = str(i)
            id2label[str(i)] = label

        ##############################################################
        # Preprocessing
        ##############################################################
        pbar.update(30)
        pbar.set_description("Load Image Processor")
        feature_extractor, transforms = get_preprocessors()
        dataset = dataset.with_transform(transforms)

        data_collator = DefaultDataCollator()

        ##############################################################
        # Load Model
        ##############################################################
        pbar.update(40)
        pbar.set_description("Load Model")
        model = AutoModelForImageClassification.from_pretrained(
            "google/vit-base-patch16-224-in21k",
            num_labels=len(labels),
            id2label=id2label,
            label2id=label2id,
        )

        ##############################################################
        # Train Model
        ##############################################################
        pbar.update(50)
        pbar.set_description("Training Model")

        training_args = TrainingArguments(
            output_dir=f"{ROOT}results",
            evaluation_strategy="epoch",
            # save_strategy="epoch",
            save_strategy="no",
            num_train_epochs=2,
            fp16=False,
            logging_steps=10,
            # save_total_limit=2,
            remove_unused_columns=False,
        )

        for batch in range(4):

            # Calculate the desired learning rate and batch size based on the current epoch
            current_learning_rate = calculate_learning_rate(batch)
            current_batch_size = calculate_batch_size(batch)

            training_args.set_dataloader(train_batch_size=current_batch_size)
            training_args.set_optimizer(learning_rate=current_learning_rate)

            trainer = Trainer(
                model=model,
                args=training_args,
                data_collator=data_collator,
                train_dataset=dataset["train"],
                eval_dataset=dataset["validation"],
                tokenizer=feature_extractor,
            )

            trainer.train()

        ##############################################################
        # Test Model
        ##############################################################
        pbar.update(80)
        pbar.set_description("Getting Predictions")
        predictions = trainer.predict(dataset['test'])
        y_pred = [np.argmax(p) for p in predictions[0]]
        y_true = [point['label'] for point in dataset['test']]
        ##############################################################
        # Calculate Metrics
        ##############################################################
        pbar.update(90)
        pbar.set_description("Calculating Metrics")
        # Confusion Matrix
        confusion_matrix = confusion_matrix(y_true, y_pred)
        cm_display = ConfusionMatrixDisplay(
            confusion_matrix=confusion_matrix, display_labels=labels)
        cm_display.plot()
        plt.title("Confusion Matrix")
        plt.show()

        # Calculate evaluation metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average=None)
        recall = recall_score(y_true, y_pred, average=None)
        f1 = f1_score(y_true, y_pred, average=None)

        # Print the evaluation metrics
        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1-score:", f1)

        ##############################################################
        # Save Model
        ##############################################################
        pbar.update(100)
        pbar.set_description("Saving Models")
        trainer.save_model(f'{ROOT}trained_model')

        # Cleanup
        if os.path.exists(EXTRACTED_IMAGES):
            shutil.rmtree(EXTRACTED_IMAGES)
