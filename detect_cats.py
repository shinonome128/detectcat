
"""
必要モジュールロード
"""
import pickle
import sys
# from skimage import io, feature, color
# transform.rescale() 関数が必要
from skimage import io, feature, color, transform
from get_histogram import get_histogram


"""
1. 主処理

グレースケールで読み込んだ画像サイズは行 500, 幅 375
(Pdb) target.shape
(500, 375)

中身を見ると、、、
(Pdb) target
array([[0.35889882, 0.36139961, 0.33388667, ..., 0.53752275, 0.53668941,
        0.53668941],
       [0.32797961, 0.37576314, 0.36448784, ..., 0.54061098, 0.54061098,
        0.53668941],
       [0.33190118, 0.33420941, 0.29275294, ..., 0.53668941, 0.53276784,
        0.53333333],
       ...,
       [0.37258392, 0.40897922, 0.38012314, ..., 0.09589373, 0.08529059,
        0.09313373],
       [0.27454471, 0.26137255, 0.2809898 , ..., 0.11127451, 0.11268118,
        0.12387333],
       [0.26388118, 0.18242824, 0.21208157, ..., 0.11571686, 0.12302431,
        0.13140314]])
(Pdb)

画像データを get_histogram() 関数に渡して得るヒストグラムは 高さ 125, 幅 93, 各要素が 26 の配列構成をとる
つまり、ウィンドウサイズ毎の LBP 特徴量を求めた結果になる
LBP 計算するときに、4 * 4 のセルサイズでやっているため、ヒストグラムの高さと幅も四分の一になっている
(Pdb) histogram.shape
(125, 93, 26)

実際の中身
(Pdb) histogram
array([[[ 2,  2,  0, ...,  0,  0,  3],
        [ 1,  0,  1, ...,  0,  0, 11],
        [ 3,  1,  0, ...,  0,  0,  8],
        ...,
    (省略)

要素の中身、これは LBP 特徴量処理した値
(Pdb) histogram[0, 0]
array([2, 2, 0, 2, 0, 1, 2, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 3])
(Pdb)

(Pdb) histogram.shape[0]
125


最初のループでとれる変数 feature の赤身、リシェープ前
入れ子配列状態で、高さ 64, 幅 64, 各要素 26 が取得できる
(Pdb) feature = histogram[y:(y + HEIGHT), x:(x + WIDTH)]
(Pdb) feature
array([[[ 2,  2,  0, ...,  0,  0,  3],
        [ 1,  0,  1, ...,  0,  0, 11],
        [ 3,  1,  0, ...,  0,  0,  8],
        ...,
        [ 1,  1,  0, ...,  0,  0,  5],
        [ 0,  0,  0, ...,  0,  0,  5],
        [ 2,  2,  0, ...,  0,  0,  4]],
        (省略)
       [[ 2,  0,  1, ...,  0,  0,  3],
        [ 2,  0,  0, ...,  0,  0,  8],
        [ 0,  1,  0, ...,  1,  2,  5],
        ...,
        [ 2,  1,  0, ...,  0,  0, 11],
        [ 2,  0,  0, ...,  0,  5,  6],
        [ 2,  4,  1, ...,  0,  0,  6]]])
(Pdb) feature.shape
(64, 64, 26)

これを -1 でリシェープすると 106496 の一直線の配列を取得できる
(1, -1)で指定したのは、svmで待ち受ける次元が違うから
*** ValueError: Expected 2D array, got 1D array instead:
array=[2 2 0 ... 0 0 6].
Reshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample.

(Pdb) feature = histogram[y:(y + HEIGHT), x:(x + WIDTH)].reshape(1, -1)
(Pdb) feature.shape
(1, 106496)
(Pdb) feature
array([[2, 2, 0, ..., 0, 0, 6]])

"""
def main():

    # バイナリ指定で学習済みモデルを開く
    svm = pickle.load(open(sys.argv[1], 'r+b'))

    # 走査対象の画像をグレースケールで読み込む
    target = color.rgb2gray(io.imread(sys.argv[2]))
    # import pdb; pdb.set_trace()

    detections = detect_cats(svm, target)
    # print(detections)


"""
検出処理部分をモジュール化
"""
def detect_cats(svm, target):

    # 検出窓サイズ、セルサイズは LBP 計算するときのセルサイズ
    WIDTH, HEIGHT = (64, 64)
    CELL_SIZE = 4
    # THRESHOLD = 3.0
    THRESHOLD = 2.5

    # ブロードキャスティングで、全要素に 0 加算、目的は不明
    target_scaled = target + 0
    # import pdb; pdb.set_trace()

    # 0.9 倍ずつ、縮小して走査
    scale_factor = 2.0 ** (-1.0 / 8.0)
    # import pdb; pdb.set_trace()

    # 検出結果を可能する空リストをあらかじめ用意
    detections = []

    # 0 ～ 16 まで s の値でループ処理、一回目は等倍処理、2回目はスケールファクタをかけて、縮小処理してゆく
    for s in range(16):
        # import pdb; pdb.set_trace()

        # 画像データから、ヒストグラムを求める
        histogram = get_histogram(target_scaled)
        # import pdb; pdb.set_trace()

        # y は 0 から セル幅(4ピクセル)ずつ、検出窓(64ピクセル)を動かしながら、ヒストグラムの高さから検出窓を引いた値(125 - 64 = 61) まで繰り返す
        # range の引数でとれるのは int 型、のみ
        # ヒストグラムの高さから検出窓分を引いて、セル幅で増加してゆく
        # for y in range(0, int((histogram.shape[0] - HEIGHT) / CELL_SIZE)):
        # for y in range(0, histogram.shape[0] - HEIGHT, CELL_SIZE):
        # 正しくはヒストグラム配列から、検出窓分引くだけでよい
        # 検出窓の単位は 64 * 64 ピクセルなので、セルサイズだと、 64 / 4 = 16 セル分、検出窓サイズを引く
        # y が一個増えると、セル一個分ずれるので、4 ピクセル動く
        for y in range(0, histogram.shape[0] - int(HEIGHT / CELL_SIZE)):
            # import pdb; pdb.set_trace()

            # x は 0 からのループ処理、考え方は同じ
            # for x in range(0, int((histogram.shape[1] - WIDTH) / CELL_SIZE)):
            # for x in range(0, histogram.shape[1] - WIDTH, CELL_SIZE):
            for x in range(0, histogram.shape[1] - int(WIDTH / CELL_SIZE)):
                # import pdb; pdb.set_trace()

                # 検出窓の内のヒストグラムを取り出し、一列に並べ、特徴ベクトルにする
                # スライシング文法は下記に従うので、
                # [i : j] で i 以上、 j 未満をスライシング
                # [i, :] で i 行目をスライシング
                # [:, j] で j 列目をスライシング
                # x軸の一回目のループ処理では [0:64, 0:64]
                # x軸の二回目のループ処理では [0:64, 4:68]
                # x軸の三回目のループ処理では [0:64, 8:72]

                # ここの元のコードが間違っている
                # feature = histogram[y:int(y + HEIGHT / CELL_SIZE), x:int(x + WIDTH / CELL_SIZE)].reshape(1, -1)
                # feature = histogram[y:(y + HEIGHT), x:(x + WIDTH)].reshape(1, -1)
                # ここも64 * 64 ピクセルの検出窓のヒストグラムを検出するのでセルサイズで割る
                # 検出窓は 16 * 16 セル
                feature = histogram[y:(y + int(HEIGHT / CELL_SIZE)), x:(x + int(WIDTH / CELL_SIZE))].reshape(1, -1)
                # import pdb; pdb.set_trace()

                # これだとうまくいかない
                # hoge = histogram[0:64, 0:64].reshape(1, -1)
                # score = svm.decision_function(hoge)
                # predict = svm.predict(hoge)

                # セルサイズで割るとうまくゆく、ヒストグラムのロジックからおう
                # hoge = histogram[0:16, 0:16].reshape(1, -1)
                # score = svm.decision_function(hoge)
                # predict = svm.predict(hoge)

                score = svm.decision_function(feature)
                # import pdb; pdb.set_trace()
                # print(s, y, x, score)

                # 検出処理
                # スコアが設定した閾値より多ければその場所に猫顔があるとみなす
                if score[0] > THRESHOLD:

                    # 画像の倍率を表示、一回目だと s = 0 なので スケールファクターのゼロ乗で 1 倍率
                    scale = (scale_factor ** s)
                    # import pdb; pdb.set_trace()

                    # 検出結果を格納する変数に矩形位置、大きさ、スコアを追加
                    detections.append({
                        'x': x * CELL_SIZE / scale,
                        'y': y * CELL_SIZE / scale,
                        'width': WIDTH / scale,
                        'height': HEIGHT / scale,
                        'score': score})

                    print(
                        y * CELL_SIZE / scale,
                        x * CELL_SIZE / scale,
                        HEIGHT / scale,
                        WIDTH / scale,
                        score)

                # import pdb; pdb.set_trace()
                # print(s, y, x)

        # transform.rescale() で元画像をスケールファクター倍、つまり 0.9 倍する
        print(s, target_scaled.shape, len(detections))
        target_scaled = transform.rescale(target_scaled, scale_factor)
        # import pdb; pdb.set_trace()

  
    # 検出窓サイズ、セルサイズは LBP 計算するときのセルサイズ
    return detections

"""
お作法、他ファイルから呼び出された場合は、このスクリプトは実行されない
"""
if __name__ == "__main__":
    main()
