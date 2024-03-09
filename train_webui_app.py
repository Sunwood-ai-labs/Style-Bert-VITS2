import streamlit as st
import os
import yaml
from webui_train import preprocess_all, get_path

def setup_environment():
    """環境構築を行う関数"""
    st.write("環境構築中...")
    os.system("git clone https://github.com/litagin02/Style-Bert-VITS2.git")
    os.chdir("Style-Bert-VITS2/")
    os.system("pip install -r requirements.txt")
    os.system("apt install libcublas11")
    os.system("python initialize.py --skip_jvnv")
    st.write("環境構築が完了しました。")

def setup_paths():
    """パスの設定を行う関数"""
    dataset_root = st.text_input("学習に必要なファイルや途中経過が保存されるディレクトリ", "/Data")
    assets_root = st.text_input("学習結果（音声合成に必要なファイルたち）が保存されるディレクトリ", "/model_assets")

    with open("configs/paths.yml", "w", encoding="utf-8") as f:
        yaml.dump({"dataset_root": dataset_root, "assets_root": assets_root}, f)

    os.makedirs(dataset_root, exist_ok=True)

def preprocess_data():
    """前処理を行う関数"""
    model_name = st.text_input("モデルの名前", "zundamon")
    use_jp_extra = st.checkbox("JP-Extra（日本語特化版）を使う", False)
    batch_size = st.number_input("学習のバッチサイズ", value=4, min_value=1, step=1)
    epochs = st.number_input("学習のエポック数", value=100, min_value=1, step=1)
    save_every_steps = st.number_input("保存頻度（何ステップごとにモデルを保存するか）", value=1000, min_value=1, step=1)
    normalize = st.checkbox("音声ファイルの音量を正規化する", True)
    trim = st.checkbox("音声ファイルの開始・終了にある無音区間を削除する", True)
    yomi_error = st.selectbox("読みのエラーが出た場合にどうするか", ("raise", "skip", "use"))

    if st.button("前処理を開始"):
        preprocess_all(
            model_name=model_name,
            batch_size=batch_size,
            epochs=epochs,
            save_every_steps=save_every_steps,
            num_processes=2,
            normalize=normalize,
            trim=trim,
            freeze_EN_bert=False,
            freeze_JP_bert=False,
            freeze_ZH_bert=False,
            freeze_style=False,
            freeze_decoder=False,
            use_jp_extra=use_jp_extra,
            val_per_lang=0,
            log_interval=200,
            yomi_error=yomi_error
        )
        st.write("前処理が完了しました。")

def train_model():
    """モデルの学習を行う関数"""
    model_name = st.text_input("学習するモデル名", "zundamon")
    use_jp_extra = st.checkbox("日本語特化版を使う", False)

    dataset_path, _, _, _, config_path = get_path(model_name)

    with open("default_config.yml", "r", encoding="utf-8") as f:
        yml_data = yaml.safe_load(f)
    yml_data["model_name"] = model_name
    with open("config.yml", "w", encoding="utf-8") as f:
        yaml.dump(yml_data, f, allow_unicode=True)

    if st.button("学習を開始"):
        if use_jp_extra:
            os.system(f"python train_ms_jp_extra.py --config {config_path} --model {dataset_path} --assets_root {assets_root}")
        else:
            os.system(f"python train_ms.py --config {config_path} --model {dataset_path} --assets_root {assets_root}")
        st.write("学習が完了しました。")

def test_results():
    """学習結果を試す関数"""
    if st.button("学習結果を試す"):
        os.system(f"python app.py --share --dir {assets_root}")

def style_vectors():
    """スタイルベクトルを生成する関数"""
    if st.button("スタイルベクトルを生成"):
        os.system("python webui_style_vectors.py --share")
        st.write("スタイルベクトルの生成が完了しました。")

def merge_models():
    """モデルをマージする関数"""
    if st.button("モデルをマージ"):
        os.system("python webui_merge.py --share")
        st.write("モデルのマージが完了しました。")

def main():
    st.title("Style-Bert-VITS2 Streamlit App")

    st.header("0. 環境構築")
    # setup_environment()

    st.header("1. 初期設定")
    setup_paths()

    st.header("3. 学習の前処理")
    preprocess_data()

    st.header("4. 学習")
    train_model()

    st.header("学習結果を試す")
    test_results()

    st.header("5. スタイル分け")
    style_vectors()

    st.header("6. マージ")
    merge_models()

if __name__ == "__main__":
    main()