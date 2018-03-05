from sklearn.metrics import confusion_matrix, f1_score

labels = [2, 0, 2, 2, 0, 1]
pred = [2, 0, 2, 2, 0, 1]

f1_score = f1_score(labels, pred, average=None  )
print("F1 Score: ", f1_score)

confusion = confusion_matrix(labels, pred)
print("Confusion: ")
print(confusion)
