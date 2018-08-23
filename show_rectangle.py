"""
モジュールロード
"""
import sys
# import numpy as np
from skimage import io
from skimage.transform import rescale
import matplotlib.pyplot as plt
import matplotlib.cm as cm


"""
モジュールテスト用途
"""
def main():

    # 画像は一つでよいので、テンプレート ax1 は無効化
    # template_path = sys.argv[1]
    # import pdb; pdb.set_trace()

    # target_path = sys.argv[2]
    target_path = sys.argv[1]
    # import pdb; pdb.set_trace()

    # 画像は一つでよいので、テンプレート ax1 は無効化
    # template = io.imread(template_path, as_grey = True)
    # import pdb; pdb.set_trace()
    # io.imshow(template)
    # io.show()

    target = io.imread(target_path, as_grey = True)
    # import pdb; pdb.set_trace()
    # io.imshow(target)
    # io.show()

    # 0.9 倍ずつ、縮小して走査
    scale_factor = 2.0 ** (-1.0 / 8.0)

    # プロット時に使うので一応書いておく
    # 検出窓サイズ、セルサイズは LBP 計算するときのセルサイズ
    WIDTH, HEIGHT = (64, 64)

    # テストパラメータ
    s = 4
    y = 228
    x = 197

    # 矩形表示
    # show_rectangle(target, scale_factor, s, y, x)
    show_rectangle(target, scale_factor, s, y, x, HEIGHT, WIDTH)


"""
結果を可視化
インプット画像データ: target
スケールファクタ: scale_factor
縮小した回数: s
検出時の座標: y, x
検出窓サイズ: HEIGHT, WIDTH
"""
# def show_rectangle(target, scale_factor, s, y, x):
def show_rectangle(target, scale_factor, s, y, x, HEIGHT, WIDTH):

    # 画像は一つでよいので、テンプレート ax1 は無効化
    # 可視化ウィンドウの中の画像を定義
    # fig, (ax1, ax2) = plt.subplots(ncols = 2, figsize = (8, 3))
    fig, (ax2) = plt.subplots(ncols = 1, figsize = (8, 3))

    # 画像は一つでよいので、テンプレート ax1 は無効化
    # ax1 を定義、テンプレート画像をグレースケールで表示
    # ax1.imshow(template, cmap = cm.Greys_r)

    # 画像は一つでよいので、テンプレート ax1 は無効化
    # 画像の軸を表示しない
    # ax1.set_axis_off()

    # 画像は一つでよいので、テンプレート ax1 は無効化
    # タイトルを設定
    # ax1.set_title('template')

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
    # th, tw = template.shape

    # 矩形を定義
    # rect = plt.Rectangle((y / scale, x / scale), tw / scale, th / scale, edgecolor = 'r', facecolor = 'none')

    print(target.shape)
    print(y, x)
    print(s)
    print(scale_factor)
    print(scale)
    # import pdb; pdb.set_trace()
    # rect = plt.Rectangle((y , x), 64, 64, edgecolor = 'r', facecolor = 'none')
    # rect = plt.Rectangle((y / scale, x / scale), 64 / scale, 64 / scale , edgecolor = 'r', facecolor = 'none')
    rect = plt.Rectangle((y, x), HEIGHT / scale, WIDTH / scale , edgecolor = 'r', facecolor = 'none')

    # マッチングした部分を矩形で囲う
    ax2.add_patch(rect)

    # 表示
    plt.show()
    # import pdb; pdb.set_trace()

    return

"""
お作法ね
直接実行時は __main__ モジュールとして実行、スクリプトから呼び出すと、実行しない
テスト作成中やモジュール化したときに使わせたくない場合に書いておく
"""
if __name__ == "__main__":
    main()
