import argparse
from pathlib import Path

import joblib
import pandas as pd


def run_inference(input_csv: str, output_csv: str, model_path: str, scaler_path: str) -> None:
    input_path = Path(input_csv)
    output_path = Path(output_csv)

    if not input_path.exists():
        raise FileNotFoundError(f"Arquivo de entrada nao encontrado: {input_path}")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    X = pd.read_csv(input_path)
    X_scaled = scaler.transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

    prob_maligno = model.predict_proba(X_scaled_df)[:, 1]
    pred = (prob_maligno >= 0.5).astype(int)

    result = X.copy()
    result["predicao"] = pred
    result["prob_maligno"] = prob_maligno

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)

    print(f"Predicoes salvas em: {output_path}")
    print("Legenda predicao: 1=Maligno, 0=Benigno")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inferencia com modelo final oficial")
    parser.add_argument("--input", required=True, help="CSV com features de entrada")
    parser.add_argument("--output", required=True, help="CSV de saida com predicoes")
    parser.add_argument(
        "--model",
        default="outputs/models/modelo_final_oficial.pkl",
        help="Caminho do modelo .pkl",
    )
    parser.add_argument(
        "--scaler",
        default="outputs/models/scaler_final_oficial.pkl",
        help="Caminho do scaler .pkl",
    )

    args = parser.parse_args()
    run_inference(args.input, args.output, args.model, args.scaler)


if __name__ == "__main__":
    main()
