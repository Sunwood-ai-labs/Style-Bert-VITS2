import streamlit as st
import os
import yaml
from webui_train import preprocess_all, get_path
import initialize
import threading

def process_transcript(dataset_root, model_name):
    # ファイルを読み込む
    with open(dataset_root + f"/{model_name}/recitation_transcript_utf8.txt", "r", encoding="utf-8") as file:
        lines = file.readlines()

    # コンバート
    result = []
    for line in lines:
        strs = line.split(",")[0].split(":")
        result.append("recitation" + strs[0][-3:] + ".wav|" + model_name + "|JP|" + strs[1] + "\n")

    # 変更をファイルに保存
    with open(dataset_root + f"/{model_name}/esd.list", "w", encoding="utf-8") as file:
        file.writelines(result)

    # 去々年を含まない行だけを選択
    cleaned_result = [line for line in result if "去々年" not in line]

    # ファイルパスを指定
    file_path = dataset_root + f"/{model_name}/esd.list"

    # ファイルに変更を保存
    with open(file_path, "w", encoding="utf-8") as file:
        file.writelines(cleaned_result)

def main():
    st.title("Style-Bert-VITS2 学習アプリ")

    # 初期設定
    st.header("1. 初期設定")
    dataset_root = st.text_input("学習データが保存されるディレクトリ", "/Data")
    assets_root = st.text_input("学習結果が保存されるディレクトリ", "/model_assets")

    if st.button("初期設定を開始"):
        with st.spinner("初期設定中..."):
            # プログレスバーを表示
            progress_bar = st.progress(0)
            initialize.main(dataset_root=dataset_root, assets_root=assets_root)
            progress_bar.progress(100)
        
            st.success("初期設定が完了しました。")
    

        with open("configs/paths.yml", "w", encoding="utf-8") as f:
            yaml.dump({"dataset_root": dataset_root, "assets_root": assets_root}, f)
        st.success("初期設定が保存されました。")

    # 学習の前処理
    st.header("2. 学習の前処理")
    model_name = st.text_input("モデルの名前", "zundamon")
    use_jp_extra = st.checkbox("JP-Extra（日本語特化版）を使う", False)
    batch_size = st.number_input("バッチサイズ", value=4, min_value=1, max_value=16)
    epochs = st.number_input("エポック数", value=100, min_value=1)
    save_every_steps = st.number_input("保存頻度（ステップ数）", value=1000, min_value=100)
    normalize = st.checkbox("音声ファイルの音量を正規化する", True)
    trim = st.checkbox("音声ファイルの無音区間を削除する", True)
    yomi_error_options = {"raise": "中断", "skip": "スキップ", "use": "無理やり使う"}
    yomi_error = st.selectbox("読みのエラー時の処理", list(yomi_error_options.keys()), format_func=lambda x: yomi_error_options[x])

    if st.button("学習の前処理を実行"):
        process_transcript(dataset_root, model_name)
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
        st.success("学習の前処理が完了しました。")

    # 学習
    st.header("3. 学習")
    if st.button("学習を開始"):
        dataset_path, _, _, _, config_path = get_path(model_name)

        with open("default_config.yml", "r", encoding="utf-8") as f:
            yml_data = yaml.safe_load(f)
        yml_data["model_name"] = model_name
        with open("config.yml", "w", encoding="utf-8") as f:
            yaml.dump(yml_data, f, allow_unicode=True)

        if use_jp_extra:
            os.system(f"python train_ms_jp_extra.py --config {config_path} --model {dataset_path} --assets_root {assets_root}")
        else:
            os.system(f"python train_ms.py --config {config_path} --model {dataset_path} --assets_root {assets_root}")
        
        st.success("学習が完了しました。")

if __name__ == "__main__":
    main()