cnnでnc比を予測します。

ファイル説明
batch.py バッチによる確率的勾配降下法を行うためのスクリプトです。
libs.py cnnの実装に使う関数群です
prepare_data.py 画像を扱いやすいデータに変換します。
process_data.py prepare_dataで生成されたデータを読み込む際に使います。
README.md このファイルです。
settings.yml 訓練時のデータ数などを変更できます。
train.py このスクリプトを走らせるとprocess_dataでデータを読み込み、cnnによる学習を行います。

cell_data 細胞の画像です。
data prepare_dataで生成されたデータはここに入ります。
saver cnnの変数をsaver内のtmpに保存します。


使い方
1.cell_dataに画像を入れます。ファイルは細胞ごとに分けてもいいです。
2.コマンドプロンプトでcell_cnnに移動し、"python prepare_data.py"でprepare_dataを走らせます。これによって下準備ができました。
3."python train.py"で学習が始まります。



結果
全然うまく予想できてません。lossは正解のnc比と予想のnc比の二乗誤差の平均ですが、0.02くらいでかなり高いです。最後に表示されるのは10段階ごとの正答率ですが、これも低いです。
