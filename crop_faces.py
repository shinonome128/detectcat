# モジュールロード
import sys
import numpy as np
from skimage import io, transform
from glob import iglob, glob


"""
1. 主処理
"""
def main():

    # 引数チェック、sys.argv は本体と引数のリスト、引数が足りない場合は、警告文を表示し、処理から抜ける
    if len(sys.argv) < 3:

        # 警告文
        print ("./crop.faces.py INPUT_DIR OUTPUT_DIR")

        # ループ処理から抜ける
        return

    # 第1引数をを変数 input_dir に格納、第2引数を変数 output_dir に格納
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    # import pdb; pdb.set_trace()

    # enumerate(リスト) でインデクス番号と要素をセットで取り出す
    # glob() で全てのjpgファイルを相対パス形式でリスト化
    # iglob() は glob()と動作が同じ、ただし全メモリに展開しない、処理後、破棄するので、処理が速い
    # i = 0 から enumerate() で取り出したリストの要素数まで繰り返す
    # i にはインデクス番号、 image_path にはファイル名の相対パス名が入る
    for i, image_path in enumerate(iglob('%s/*/*.jpg' % input_dir)):

        # アノテーションファイル名を相対パスで保存
        annotation_path = '%s.cat' % image_path
        # import pdb; pdb.set_trace()

        # まずは処理内容を記載
        try:
            # anotation_path ファイルを開いて、を関数 parse_anotation() に渡す、結果を変数 anotation に格納
            annotation = parse_annotation(open(annotation_path).read())

        # try 処理の例外、continue で以降の処理を中断しループ処理に戻る
        except:
            continue

        # image_path を開いて、変数 annnotation を関数 crop_face() に渡す、結果を変数 facee に格納
        face = crop_face(io.imread(image_path), annotation)
        # import pdb; pdb.set_trace()

        # 変数 face が空でない場合はディレクトリに変数 i .png で保存
        # ここでエラー発生、None の比較に ==, != を使うのではなく、 is, is not を利用する
        # if face != None:
        if face is not  None:
            # ここでエラーが発生、ウィンドウズパス表現に変更
            # io.imsave('%s/%d.png' % (output_dir, i), face)
            io.imsave('%s\%d.png' % (output_dir, i), face)


"""
2. アノテーション情報を解析
アノテーション情報を入力、入れ子配列で部位とその座標を出力

下記のような情報が渡され、変化してゆく
print(line)
9 175 160 239 162 199 199 149 121 137 78 166 93 281 101 312 96 296 133

print(line.split())
['9', '175', '160', '239', '162', '199', '199', '149', '121', '137', '78', '166', '93', '281', '101', '312', '96', '296', '133']

print(list(v))
[9, 175, 160, 239, 162, 199, 199, 149, 121, 137, 78, 166, 93, 281, 101, 312, 96, 296, 133]

print(ret)
{'left_eye': array([175, 160]), 'right_eye': array([239, 162]), 'mouth': array([199, 199]), 'left_ear1': array([149, 121]), 'left_ear2': array([137,  78]), 'left_ear3': array([166,  93]), 'right_ear1': array([281, 101]), 'right_ear2': array([312,  96]), 'right_ear3': array([296, 133])}
"""
def parse_annotation(line):
    # import pdb; pdb.set_trace()

    # line.split() で文字列 line をスプリットして、 map() で全要素を数値型(int)にして、変数v 配列として格納
    # このままだと、後続のループ処理がまわらない
    # v = map(int, line.split())
    # 一度、map オフジェクトから数値を取り出すと v を参照できないので配列で定義
    v = list(map(int, line.split()))
    # import pdb; pdb.set_trace()

    # 空配列 ret と 文字列の配列 parts を用意
    ret = {}
    parts = ["left_eye", "right_eye", "mouth",
             "left_ear1", "left_ear2", "left_ear3",
             "right_ear1", "right_ear2", "right_ear3"]
    # import pdb; pdb.set_trace()

    # enumerate() で i = 0 から 要素の数である i = 18 まで、インデクス番号と要素を取り出す
    for i, part in enumerate(parts):
        # import pdb; pdb.set_trace()

        # i が 9 以上の場合、for ループを抜ける
        if i >= v[0]: break
        # import pdb; pdb.set_trace()

        # 入れ子配列の作成、各要素の y 軸と x 座業を作ってゆく
        ret[part] = np.array([v[1 + 2 * i], v[1 + 2 * i + 1]])
        # import pdb; pdb.set_trace()

    # import pdb; pdb.set_trace()
    return ret


"""
3. 画像切り取り処理
イメージとアノテーション座標情報を入力、切り取られたイメージを出力

(Pdb) an
{'left_eye': array([175, 160]), 'right_eye': array([239, 162]), 'mouth': array([199, 199]), 'left_ear1': array([149, 121]), 'left_ear2': array([137,  78]), 'left_ear3': array([166,  93]), 'right_ear1': array([281, 101]), 'right_ear2': array([312,  96]), 'right_ear3': array([296, 133])}

(Pdb) diff_eyes
array([-64,  -2])

(Pdb) center
array([204.33333333, 173.66666667])

(Pdb) radius
70.43436661176133

(Pdb) xu
133.898966721572
(Pdb) xl
274.7676999450947
(Pdb) yu
103.23230005490532
(Pdb) yl
244.101033278428

(Pdb) image.shape
(500, 375, 3)

"""
def crop_face(image, an):
    # import pdb; pdb.set_trace()

    # 変数 diff_eyes に left_eye と right_eye の差 をとり、両目の距離を (x ,y) で格納
    diff_eyes = an["left_eye"] - an["right_eye"]
    # import pdb; pdb.set_trace()

    # 横向き画像を排除する処理
    # 両目の x 距離が 0ピクセル の場合は処理しない
    # つまり、両目の x 距離が同じ画像は横向きとみなす
    # 両目の (y 距離) / (x 距離) を演算して、0.5 ピクセル以上は処理しない
    # つまり、両目の y 距離が大きい画像は横向きとみなす
    # diff_eyes の 列座標差分を行座標差分で割って、float() で float 型に変換し、abs() で絶対値に変換
    if diff_eyes[0] == 0 or abs(float(diff_eyes[1]) / diff_eyes[0]) > 0.5:
        return None
    # import pdb; pdb.set_trace()

    # 変数 center に左目、口、右目の中央座標を格納
    center = (an["left_eye"] + an["right_eye"] + an["mouth"]) / 3
    # import pdb; pdb.set_trace()

    # 上下逆転画像を排除する処理
    # 目と口の中央値の y  座標が、口の y 座標より大きい場合は処理しない
    if center[1] > an["mouth"][1]: 
        return None
    # import pdb; pdb.set_trace()

    # いままで両目の距離は右目と左目の座標差分で種痘していたので、正確な距離を算出
    # 変数 radius (半径) に diff_eyes 座標までの長さの 1.1 倍した値を入れる
    radius = np.linalg.norm(diff_eyes) * 1.1
    # import pdb; pdb.set_trace()

    # 左目、口、右目の中央座業を中心に、両目の距離で半径を描いて、 x 軸, y 軸の上限下限値を算出
    xu = center[0] - radius
    xl = center[0] + radius
    yu = center[1] - radius
    yl = center[1] + radius
    # import pdb; pdb.set_trace()

    # 左目、口、右目の中央座業を中心に、両目の距離で半径を描いて、見切れる画像は処理しない
    # 左側、上側に見切れている場合の条件式
    if xu < 0 or yu < 0: 
        return None

    # 右側、下側に見切れている場合の条件式
    # image.shape で 画像サイズを取得、縦、横、属性の順番
    if xl > image.shape[1] or yl > image.shape[0]:
        return None

    # ここでエラーが発生、スライシングは int 型でなければならいので変換すること
    # cropped = image[yu:yl, xu:xl]
    cropped = image[int(yu):int(yl), int(xu):int(xl)]
    # import pdb; pdb.set_trace()

    # 最後にリサイズ
    return transform.resize(cropped, (64, 64))


"""
お作法、他ファイルから呼び出された場合は、このスクリプトは実行されない
"""
if __name__ == "__main__":
    main()
