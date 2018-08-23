
"""
必要モジュールロード
"""
import pickle
import sys
from skimage import io, feature, color, transform
from detect_cats import detect_cats
# import numpy as np


"""
主処理部分

今回はモジュールだけを開発したいので主処理には、開発済みのモジュールだけを記載
detect_catアウトプットを変数で渡す、やり方は
1. バイナリから呼び出す、
2. モジュールでdetect_catを呼び出して結果を受け取る
一番楽な、モジュールを呼び出す方法で実装する
"""
def main():

    # バイナリ指定で学習済みモデルを開く
    svm = pickle.load(open(sys.argv[1], 'r+b'))

    # 走査対象の画像をグレースケールで読み込む
    target = color.rgb2gray(io.imread(sys.argv[2]))
    # import pdb; pdb.set_trace()

    # 猫顔検出し座標、矩形情報を得る
    detections = detect_cats(svm, target)
    # print(detections)
    # print(type(detections))
    # import pdb; pdb.set_trace()

    # NMS を適用
    detections_nms = apply_nms(detections)
    print(detections_nms)


"""
1. 二つの検出矩形の重なり度を計算する
"""
def overlap_score(a, b):
    # import pdb; pdb.set_trace()

    # left = max(a['x'], b['x'])
    # left = max(print(a['x']), print(b['x']))
    # left = max(float(print(a['x'])), float(print(b['x'])))
    left = max(float(a['x']), float(b['x']))
    # import pdb; pdb.set_trace()
    
    # right = min(a['x'], x['width'], b['x'] + b['width'])
    right = min(float(a['x']), float(a['width']), float(b['x']) + float(b['width']))

    # top = max(a['y'], b['y'])
    top = max(float(a['y']), float(b['y']))

    # bottom = min(a['y'] + a['height'], b['y'] + b['height'])
    bottom = min(float(a['y']) + float(a['height']), float(b['y']) + float(b['height']))

    intersect = max(0, (right - left) * (bottom - top))

    # union = a['width'] * a['height'] + b['width'] * b['height'] - intersect
    union = float(a['width']) * float(a['height']) + float(b['width']) * float(b['height']) - intersect

    return intersect / union


"""
2. NMS適用処理

"""
def apply_nms(detections):
    # print(type(detections))
    # import pdb; pdb.set_trace()

    # 検出矩形をスコアで降順に並べ替える
    # detections = sorted(detectons, key = lambda d: d['score'], reverse = True)
    detections = sorted(detections, key = lambda d: d['score'], reverse = True)
    # print(type(detections))
    # import pdb; pdb.set_trace()

    deleted = set()

    # スコアの大きいものから順番に見ていく
    for i in range(len(detections)):

        # 各検出矩形は自身よりスコアの小さい矩形のうち重なり度が閾値 (0.3) より大きい物を探し、該当する検出矩形を削除
        if i in deleted: continue

        for j in range(i + 1, len(detections)):
            # print(type(detections[i]))
            # print(type(detections[j]))
            # import pdb; pdb.set_trace()

            # print(overlap_score(detections[i], detections[j]))
            if overlap_score(detections[i], detections[j]) > 0.3:

                deleted.add(j)

    detections = [d for i, d in enumerate(detections) if not i in deleted]

    return detections

"""
お作法、他ファイルから呼び出された場合は、このスクリプトは実行されない
"""
if __name__ == "__main__":
    main()
