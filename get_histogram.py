
"""
必要モジュールのロード
"""
import sys
import numpy as np
from skimage import io, feature


"""
1. 主処理
"""
def main():

    # 引数チェック、sys.argv は本体と引数のリスト、引数が足りない場合は、警告文を表示し、処理から抜ける
    if len(sys.argv) < 2:

        # 警告文
        print ("./tmp1.py IMAGE_PATH")

        # ループ処理から抜ける
        return

    # 変数 image_path に引数を格納
    image_path = sys.argv[1]

    # イメージからヒストグラムを求める関数
    # LBP計算は輝度計算なのでグレースケールで読み込む
    histogram = get_histogram(io.imread(image_path, as_grey = True))
    # import pdb; pdb.set_trace()

    # セル毎のヒストグラムを最後に一列に並べる、三次元配列を一次元配列、つまりベクトルにする
    feature_vector = histogram.reshape(-1)
    # import pdb; pdb.set_trace()


"""
2. セルごとのLBP特徴量のヒストグラムを求める関数

読み込んだ画像の形
(Pdb) image.shape
(64, 64)

LBP特徴計算結果の形
(Pdb) lbp.shape
(64, 64)

読み込んだ画像の 0 行目
(Pdb) image[0]
array([0.65464902, 0.77062941, 0.87903451, 0.83053137, 0.79133843,
       0.70696078, 0.59170863, 0.57654314, 0.55579647, 0.55750784,
       0.5956302 , 0.64190824, 0.65958157, 0.61506039, 0.64645569,
       0.65908353, 0.59605569, 0.62996588, 0.69719804, 0.7641702 ,
       0.74679451, 0.70030118, 0.73082549, 0.7736949 , 0.72970941,
       0.74092353, 0.78406078, 0.74876667, 0.75634196, 0.71011647,
       0.70983373, 0.74515059, 0.83984863, 0.77987137, 0.79218667,
       0.81654941, 0.86645137, 0.82946784, 0.82329137, 0.83254118,
       0.82666235, 0.84318196, 0.85665098, 0.81939255, 0.79949412,
       0.82975059, 0.80620627, 0.75099882, 0.85690392, 0.84909843,
       0.86868353, 0.84907569, 0.79584039, 0.88433216, 0.94067843,
       0.88857412, 0.91377804, 0.91545961, 0.91688118, 0.85470157,
       0.90091216, 0.91912118, 0.8841098 , 0.87904941])

LBPの 0 行目
(Pdb) lbp[0]
array([ 2.,  1.,  0.,  0.,  0., 25., 25., 13., 13., 13., 12., 10.,  8.,
       11.,  8., 13., 13., 13., 12., 25., 10., 12., 13., 25., 12., 12.,
       25., 25., 25., 25., 25., 25.,  0.,  3.,  2., 25.,  0.,  0.,  2.,
       25.,  1.,  0.,  0., 25., 25., 25., 25., 25.,  1.,  3., 25., 25.,
       25., 25.,  0.,  1.,  0.,  1.,  1., 25.,  1.,  0.,  0.,  1.])

histogram 用の配列サイズ、16 * 16 の多次元配列
(Pdb) histogram.shape
(16, 16, 26)

ひとつの要素が 長さ 16 の配列、セル (0, 0) のヒストグラム
(Pdb) histogram[0, 0]
array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0])

ループ処理完了後のセル (0, 0) のヒストグラム
(Pdb) histogram[0, 0]
array([4, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 7])

ヒストグラムループでセルのヒストグラムの値の変化
セル内で dy, dx をインクリメントしながら、LBP特徴量をプロットしてゆく

LBP特徴量が 2 のとき、2 のビンに加算
y x dy dx dbp:  0 0 0 0 2
bfr:  [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
afr:  [0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]

LBP特徴量が 1 のとき、1 のビンに加算
y x dy dx dbp:  0 0 0 1 1
bfr:  [0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
afr:  [0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]

LBP特徴量が 0 のとき、0 のビンに加算
y x dy dx dbp:  0 0 0 2 0
bfr:  [0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
afr:  [1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]

LBP特徴量が 0 のとき、0 のビンに加算
y x dy dx dbp:  0 0 0 3 0
bfr:  [1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
afr:  [2 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
"""
def get_histogram(image):

    # 近傍 24 ピクセル、半径 3 でLBP計算
    LBP_POINTS = 24
    LBP_RADIUS = 3

    # セルサイズ 4 * 4 ピクセルの LBP ヒストグラム
    CELL_SIZE = 4

    # エラーが出てる
    # ValueError: The parameter `image` must be a 2-dimensional array
    # LBP特徴利用時はグレースケールで読み込む必要がある
    lbp = feature.local_binary_pattern(image, LBP_POINTS, LBP_RADIUS, 'uniform')
    # import pdb; pdb.set_trace()

    # LBP特徴量の種類は 0 - 24 の 25 通りと、non-uniform の 1 通りなので ビン数は 26 になる
    bins = LBP_POINTS + 2
    # import pdb; pdb.set_trace()

    # 変数 histogram に要素が全てゼロの配列を作成、大きさは (16, 16, 26)
    # 各セルのビンの出現回数を格納、 histgram[y, x] がセル(x, y) のヒストグラムになる
    # エラー出てる
    # TypeError: 'float' object cannot be interpreted as an integer
    # image.shape / CEL_SIZE の計算結果が float 型なので int 型に修正
    # histogram = np.zeros(shape = (image.shape[0] / CELL_SIZE, image.shape[1] / CELL_SIZE, bins), dtype = np.int)
    histogram = np.zeros(shape = (int(image.shape[0] / CELL_SIZE), int(image.shape[1] / CELL_SIZE), bins), dtype = np.int)
    # import pdb; pdb.set_trace()

    # y 軸をセル毎にずらすループ
    # y = 0 から 59 まで、4 づつ増加させながらループ処理
    # for y in range(0, image.shape[0] - CELL_SIZE, CELL_SIZE):
    for y in range(0, int(image.shape[0] - CELL_SIZE), CELL_SIZE):
        
        # x 軸をセル毎にずらすループ
        # x = 0 から 59 まで、4 づつ増加させながらループ処理
        # for x in range(0, image.shape[1] - CELLSIZE, CELL_SIZE):
        for x in range(0, int(image.shape[1] - CELL_SIZE), CELL_SIZE):

            # セルの中の y 軸をずらすループ
            # dy = 0 から 3 まで ループ処理
            for dy in range(CELL_SIZE):

                # セルの中の x 軸をずらすループ
                # dx = 0 から 3 まで ループ処理
                for dx in range(CELL_SIZE):
                    # import pdb; pdb.set_trace()

                    # 該当箇所のLBP 特徴量をスライシング、該当するセル内のヒストグラムの対応するビンに加算
                    # print("y x dy dx dbp: ", x, y, dy, dx, int(lbp[y + dy, x + dx]))
                    # print("bfr: ", histogram[int(y / CELL_SIZE), int(x / CELL_SIZE)])
                    # histogram[y / CELL_SIZE, x / CELL_SIZE, int(lbp[y + dy, x + dx])] +=1
                    histogram[int(y / CELL_SIZE), int(x / CELL_SIZE), int(lbp[y + dy, x + dx])] +=1
                    # print("afr: ", histogram[int(y / CELL_SIZE), int(x / CELL_SIZE)])
                    # import pdb; pdb.set_trace()

    # import pdb; pdb.set_trace()
    return histogram


"""
お作法、他ファイルから呼び出された場合は、このスクリプトは実行されない
"""
if __name__ == "__main__":
    main()
