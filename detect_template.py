"""
モジュールロード
"""
import sys
import numpy as np
from skimage import io
from skimage.transform import rescale
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def compute_score_map(template, target):

    """
    # (2) 拡大縮小しない場合と同じ方法で score_map を計算
    """
    th, tw = template.shape

    score_map = np.zeros(shape = (target.shape[0] - th, target.shape[1] - tw))

    # y を 0 ～スタートし、380 回繰り返す
    for y in range(score_map.shape[0]):

        # x を 0 からスタートして、342 回繰り返す
        for x in range(score_map.shape[1]):

            # ターゲット画像から、テンプレート画像サイズをを切り出し、テンプレート画像の値との差分を取得
            diff = target[y:y + th, x:x + tw] - template

            # SSD計算部分、行列diff の全要素を2条(square)して、和(sum)を求め、score_map の座業に代入
            score_map[y, x] = np.square(diff).sum()

    return score_map


def main():

    """
    # (省略) template, target をファイルから読み込み
    """
    template_path = sys.argv[1]
    # import pdb; pdb.set_trace()

    target_path = sys.argv[2]
    # import pdb; pdb.set_trace()

    template = io.imread(template_path, as_grey = True)
    # import pdb; pdb.set_trace()
    # io.imshow(template)
    # io.show()

    target = io.imread(target_path, as_grey = True)
    # import pdb; pdb.set_trace()
    # io.imshow(target)
    # io.show()


    """
    (1) 画像を 2^1/8 ずつ縮小(0.9倍)しながら各スケールの score_map を計算
    """
    # 空のリスト、 score_maps を作成
    score_maps = []
    # import pdb; pdb.set_trace()

    # 0.9倍
    scale_factor = 2.0 ** (-1.0 / 8.0)
    # import pdb; pdb.set_trace()

    target_scaled = target + 0
    # import pdb; pdb.set_trace()

    # s を 0 から 8 回繰り返す
    for s in range(8):

        # SSD計算を実施、リスト形式で各倍利率の score_map を score_maps リストにアペンドしていく
        score_maps.append(compute_score_map(template, target_scaled))
        # import pdb; pdb.set_trace()
        # print(s)
        # print(target_scaled[0, 0])

        # SDD計算対象画像に scale_factor をかけて縮小する
        target_scaled = rescale(target_scaled, scale_factor)
        # import pdb; pdb.set_trace()


    """
    (3) SSDが最小のスケール、座標を取得
    print(score)
    0.0
    print(s)
    0
    print(x, y)
    319 141
    """
    
    # s = 0 から score_maps の要素数 (score_map の数) だけループ、score_map の最小値を score に、その時の s 値、座標を (x, y) に格納
    score, s, (x, y) = min([(np.min(score_map), s,

        # agrgmin で score_map の最小値を取得し、unravel_index で、score_map のインデクスを取得
        np.unravel_index(np.argmin(score_map), score_map.shape))

        # enurate() で s = 0 から score_maps の要素数 (score_map の数) だけループ、同時に要素の score_map も取り出す
        for s, score_map in enumerate(score_maps)])

    # import pdb; pdb.set_trace()

    """
    (4) 結果を可視化
    """
    # 可視化ウィンドウの中の画像を定義
    fig, (ax1, ax2) = plt.subplots(ncols = 2, figsize = (8, 3))

    # ax1 を定義、テンプレート画像をグレースケールで表示
    ax1.imshow(template, cmap = cm.Greys_r)

    # 画像の軸を表示しない
    ax1.set_axis_off()

    # タイトルを設定
    ax1.set_title('template')

    # ax2 を定義、ターゲット画像をグレースケールで表示
    # import pdb; pdb.set_trace()
    ax2.imshow(target, cmap = cm.Greys_r)

    # 画像の軸を表示しない
    ax2.set_axis_off()

    # タイトルを設定
    ax2.set_title('target')

    # SSD値が最小時のスケールを再計算
    scale = (scale_factor ** s)

    # テンプレート画像の高さ、幅を取得
    th, tw = template.shape

    # 矩形を定義
    rect = plt.Rectangle((y / scale, x / scale), tw / scale, th / scale, edgecolor = 'r', facecolor = 'none')

    # マッチングした部分を矩形で囲う
    ax2.add_patch(rect)

    # 表示
    plt.show()


"""
お作法ね
直接実行時は __main__ モジュールとして実行、スクリプトから呼び出すと、実行しない
テスト作成中やモジュール化したときに使わせたくない場合に書いておく
"""
if __name__ == "__main__":
    main()
