
# モジュールロード
import sys
import numpy as np
from skimage import io, feature, color
from glob import iglob
import pickle
# 同一ディレクトリの get_histogram.py の中で定義された、 get_histogram() を呼ぶ
from get_histogram import get_histogram


"""
正例、負例、それぞれの特徴量を計算処理
feature の要素の増え方
1回目
[array([4, 1, 1, ..., 0, 0, 0])]
1回目の反転画像
[array([4, 1, 1, ..., 0, 0, 0]), array([2, 3, 2, ..., 0, 0, 0])]
"""
def get_features(directory):
    # 空配列を作成
    features = []

    # 変数 directory の中の PNG 形式の画像ファイル名を要素として取り出す
    for fn in iglob('%s/*.png' % directory):

        # グレースケールで画像を変数 image に格納
        image = color.rgb2gray(io.imread(fn))

        # get_histgram() 関数に画像データを渡し、画像一枚分のヒストグラムを取得し、一次元配列、つまり特徴ベクトルにする
        # append で、要素として追加、一つの要素が配列なので注意、入れ子配列
        features.append(get_histogram(image).reshape(-1))
        # import pdb; pdb.set_trace()

        # 学習データを増やすために fliplr() 関数で画像を反転させて、同じ処理を実施
        features.append(get_histogram(np.fliplr(image)).reshape(-1))
        # import pdb; pdb.set_trace()

    return features


"""
主処理
正例 2 枚、負例 2 枚の場合、フリップ処理をしているので数が 2 倍になる
(Pdb) n_positives
4
(Pdb) n_negatives
4
(Pdb)
"""
def main():

    # 引数をで正例、負例画像のディレクトリを指定
    positive_dir = sys.argv[1]
    negative_dir = sys.argv[2]

    # 正例、負例画像の特徴ベクトルを取得
    positive_samples = get_features(positive_dir)
    negative_samples = get_features(negative_dir)

    # 正例、負例画像の特徴ベクトルの数を取得
    n_positives = len(positive_samples)
    n_negatives = len(negative_samples)
    # import pdb; pdb.set_trace()

    # 二次元配列 X の前半部分に正例画像の特徴量、後半部分に負例画像の特徴量を格納
    X = np.array(positive_samples + negative_samples)

    # 配列 y に、正例に対しては 1 、負例に対しては 0 のラベルを格納
    y = np.array([1 for i in range(n_positives)] +

            [0 for i in range(n_negatives)])

   # 変数 X, y を引数で与えたファイル名で保存、pikele.dump() 関数はバイナリ保存するときに使う 
    pickle.dump((X, y), open(sys.argv[3], 'wb'))
    # import pdb; pdb.set_trace()


"""
お作法、他ファイルから呼び出された場合は、このスクリプトは実行されない
"""
if __name__ == "__main__":
    main()
