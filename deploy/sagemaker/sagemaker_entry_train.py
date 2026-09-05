import argparse
import os 
import joblib
import pandas as pd 
from sklearn.linear_model import LogisticRegression

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    args = parser.parse_args()

    train_csv_path = os.path.join(args.train, "train.csv")
    train_data = pd.read_csv(train_csv_path)

    features = train_data.drop("label", axis=1)
    target = train_data["label"]

    model = LogisticRegression()
    model.fit(features, target)

    model_path = os.path.join(args.model_dir, "model.joblib")
    joblib.dump(model, model_path)
    print(f"모델 저장 완료:{model_path}")

if __name__ == "__main__":
    main()

def model_fn(model_dir):
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    return model