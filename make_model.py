"""
必要モジュールロード
"""
import sys
import sklearn.svm
import pickle


"""
主処理
"""
def main():

    # パラメータのバイナリファイルを読み込み、説明変数 X, 目的変数y に格納
    X, y = pickle.load(open(sys.argv[1], 'r+b'))
    # import pdb; pdb.set_trace()

    # SVM のインスタンス作成
    classifier = sklearn.svm.LinearSVC(C = 0.0001)

    # インスタンスに説明変数、目的変数を食わせてモデルを構築
    classifier.fit(X, y)

    # モデルをバイナリファイルで保存
    pickle.dump(classifier, open(sys.argv[2], 'wb'))


"""
お作法、他ファイルから呼び出された場合は、このスクリプトは実行されない
"""
if __name__ == "__main__":
    main()
