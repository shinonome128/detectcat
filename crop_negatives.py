
"""
モジュールロード
"""
import sys
import numpy as np
from skimage import io, transform
from glob import glob
import random


"""
1. 主処理
"""
def main():

    # 引数チェック、サンプル画像インプットディレクトリ、負例画像アウトプットディレクトリ、生成したい負例画像数
    if len(sys.argv) < 4:

        # 警告
        # 文字列が間違っとるので修正
        # print ("./crop.faces.py INPUT_DIR OUTPUT_DIR N")
        print ("./crop_negatives.py INPUT_DIR OUTPUT_DIR N")

        # 関数から抜ける
        return

    # 引数を変数で定義
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    n_negatives = int(sys.argv[3])

    # 変数 image_list にインプットディレクトリのJPG形式とPNG形式のファイルを配列で格納
    # ここでエラー、ディレクトリの表現をウィンドウズに変更
    # image_list = glob('%s/*.jpg' % input_dir) + glob('%s/*.png' % input_dir)
    image_list = glob('%s\*.jpg' % input_dir) + glob('%s\*.png' % input_dir)
    # import pdb; pdb.set_trace()

    # カウンターをセット
    count = 0

    # i = 0 ～ 2 までループ処理、 enumerate() で image_list から インデクスと要素を取り出す
    for i, image_path in enumerate(image_list):

        # 変数 image に取り出した要素を画像で格納
        image = io.imread(image_path)
        # import pdb; pdb.set_trace()

        # 読み込んだ画像サイズの x, y の値が 64 より大きい場合は続行、条件を満たさない場合はアサート機能で強制修了
        assert image.shape[0] >= 64 or image.shape[1] >= 64
        # import pdb; pdb.set_trace()

        # 生成する負例画像数の条件式
        # 条件式を満たすと while から抜ける
        # n_negatives = 3 で サンプル画像数が 3 の場合、各画像でループは 1 回
        # 3 * 1 / 3 = 1 、つまり count = 0 で処理
        # 3 * 2 / 3 = 2 、つまり count = 1 で処理
        # 3 * 3 / 3 =`3 、つまり count = 2 で処理
        # n_negatives = 6 で サンプル画像数が 3 の場合、各画像でループは 2 回
        # 6 * 1 / 3 = 2 、つまり count = 0, 1 で処理
        # 6 * 2 / 3 = 4 、つまり count = 2, 3 で処理
        # 6 * 3 / 3 =`6 、つまり count = 4, 5 で処理
        while n_negatives * (i + 1) / len(image_list) > count:
            # import pdb; pdb.set_trace()
            
            # 関数 crop_randamly() にサンプル画像をわたして、返り値を変数 cropped に格納
            cropped = crop_randomly(image)

            # ファイルに書き出し
            io.imsave('%s/%d.png' % (output_dir, count), cropped)

            # カウンタ加算
            count += 1


"""
2. サンプル画像をランダムに切り取り、 64 ピクセルにそろえる処理
"""
def crop_randomly(image):

    # サンプル画像形状を取得
    h, w, _ = image.shape

    # randam.randint() で 0 から (画像形状最大値 - 64) の値を生成
    x = random.randint(0, w - 64)
    y = random.randint(0, h - 64)

    # ランダム生成した値から 64 ピクセルスライシングして変数 cropped に格納
    cropped = image[y:y + 64, x:x + 64]
    return cropped


"""
お作法、他モジュールから呼び出された場合は実行しない
"""
if __name__ == "__main__":
    main()
