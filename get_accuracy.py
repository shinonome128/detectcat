"""
必要モジュールロード
"""
import sys
import pickle
# import sklearn.svm


"""
主処理
"""
def main():

	# 学習済みの SVM のモデルとテスト用データの特徴ファイルから読み取り、バイナリファイル形式から読み込む
	classifier = pickle.load(open(sys.argv[1], 'r+b'))

	X, y = pickle.load(open(sys.argv[2], 'r+b'))
	
	# predict に特徴を渡してラベルを予測
	y_predict = classifier.predict(X)

	# 予測したラベルと正解を比較して正解率を求める
	correct = 0

	for i in range(len(y)):

	    if y[i] == y_predict[i]: correct += 1

	print('Accuracy: %f' % (float(correct) / len(y)))


"""
お作法、他ファイルから呼び出された場合は、このスクリプトは実行されない
"""
if __name__ == "__main__":
    main()
