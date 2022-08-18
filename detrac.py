from tools import construct_composed_dataset
from strcture import feature_extractor, feature_composer
from tools.parser import args
from pathlib import Path
import os

project_path = Path("detrac.py").parent.absolute().parent

INITIAL_DATASET_PATH = project_path.joinpath("src/data/initial_dataset")
EXTRACTED_FEATURES_PATH = project_path.joinpath("src/data/extracted_features")
COMPOSED_DATASET_PATH = project_path.joinpath("src/data/composed_dataset")
MODELS_PATH = project_path.joinpath("src/models")
MODEL_DETAILS_DIR = MODELS_PATH.joinpath(MODELS_PATH, "details")


# Training option
def training(args):

    # Get training options from argument parser
    num_epochs = args.epochs[0]
    batch_size = args.batch_size[0]
    feature_extractor_num_classes = args.num_classes[0]
    feature_composer_num_classes = 2 * feature_extractor_num_classes
    folds = args.folds[0]
    k = args.k[0]
    feature_extractor_lr = args.lr[0]
    feature_composer_lr = args.lr[1]

    # Train the feature extractor
    feature_extractor.train_feature_extractor(
        initial_dataset_path=INITIAL_DATASET_PATH,
        extracted_features_path=EXTRACTED_FEATURES_PATH,
        epochs=num_epochs,
        batch_size=batch_size,
        num_classes=feature_extractor_num_classes,
        folds=folds,
        lr=feature_extractor_lr,
        model_dir=MODELS_PATH
    )
 

    # Construct the dataset composed using the extracted features
    construct_composed_dataset.execute_decomposition(
        initial_dataset_path=INITIAL_DATASET_PATH,
        composed_dataset_path=COMPOSED_DATASET_PATH,
        features_path=EXTRACTED_FEATURES_PATH,
        k=k
    )

    # create weight for imbalanced data
    lenOfClasses = {}
    weights = {}
    for (i, class_) in enumerate(os.listdir(COMPOSED_DATASET_PATH)):
        images = os.path.join(COMPOSED_DATASET_PATH, class_)
        numOfImages = len(os.listdir(images))
        lenOfClasses[i] = numOfImages
    numOfImagesPerClass = lenOfClasses.values
    num = numOfImagesPerClass()
    largestClass = max(num)
    for (i, (j, class_)) in enumerate(zip(num, os.listdir(COMPOSED_DATASET_PATH))):
        weights[i] = j/largestClass

    # test
    for key in weights:
        print(key, '->', weights[key])

    # Train feature composer on composed dataset
    feature_composer.train_feature_composer(
        composed_dataset_path=COMPOSED_DATASET_PATH,
        epochs=num_epochs,
        batch_size=batch_size,
        num_classes=feature_composer_num_classes,
        folds=folds,
        lr=feature_composer_lr,
        model_dir=MODELS_PATH,
        class_weight=weights
    )

# Inference option
def inference(args):
    
    path_to_file = input(
        "Please enter the path of the file you wish to run the model upon (e.g.: /path/to/image.png): ")

    print(path_to_file)
    # Check if file exists
    assert os.path.exists(path_to_file)

    # Check if file is an image (no GIFs)
    assert path_to_file.lower().endswith(".png") or path_to_file.lower().endswith(".jpg") or path_to_file.lower().endswith(".jpeg")

    # Create a cache containing all trained models
    model_list = []
    print("Here is a list of your models: ")
    idx = 1
    for model in os.listdir(MODELS_PATH):
        if "feature_composer" in model:
            print(f"{idx}) {model}")
            idx += 1
            model_list.append(model)

    # Prompt user to choose a model
    model_choice = -1
    while model_choice > len(model_list) or model_choice < 1:
        model_choice = int(input(f"Which model would you like to load? [Number between 1 and {len(model_list)}]: "))

    # Predict
    prediction = feature_composer.infer(
        model_details_dir=MODEL_DETAILS_DIR,
        model_dir=MODELS_PATH,
        model_name=model_list[model_choice - 1], 
        input_image=path_to_file
    )
    print(f"Prediction: {list(prediction.keys())[0].split('_')[0]}")
    print(f"Confidence: \n{prediction}")

# Function used to initialize repo with the necessary folders.
def init_folders(path: Path):

    if not path.exists():
        print(f"{path} doesn't exist. Initializing...")
        path.mkdir()
        return True
    return False

def main():
    fresh_directories = [
        init_folders(INITIAL_DATASET_PATH),
        init_folders(EXTRACTED_FEATURES_PATH),
        init_folders(COMPOSED_DATASET_PATH),
        init_folders(MODELS_PATH),
        init_folders(MODEL_DETAILS_DIR)
    ]

    if all(fresh_directories) == True:
        print(f"The directories have just been created. Make sure to populate the {INITIAL_DATASET_PATH} with your data.")
        exit(0)
    else:
        if len(os.listdir(INITIAL_DATASET_PATH)) == 0:
            print(f"Your main data directory ({INITIAL_DATASET_PATH}) is empty. Make sure to populate it before running the script.")
            exit(0)

    init_folders(MODELS_PATH)
    
    # Mode selection.
    # If no mode is selected, exit
    if args.train == False and args.infer == False:
        # No option = No reason to use the model
        print("No option selected.")
        exit(0)

    # If one or both modes were selected
    else:
        # If both the training mode and the inference mode are selected
        if args.train == True and args.infer == True:
            print("\nPreparing the model for training and inference\n")

            # Train
            training(args)

            # Infer
            inference(args)
        else:
            # If only the training mode was selected
            if args.train == True and args.infer == False:
                print("\nPreparing the model for training\n")

                # Train
                training(args)
            # Otherwise
            elif args.train == False and args.infer == True:
                print("\nPreparing the model for inference\n")

                # Infer
                inference(args)


if __name__ == "__main__":
    main()
