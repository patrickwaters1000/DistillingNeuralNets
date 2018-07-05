import Teacher
import Student
from Utils import build_database

repeats = 1

build_database()
T = Teacher.Teacher("densenet")
T.calculate_features()
hist1 = T.train()
teacher_acc = hist1.history["val_acc"][-1]
T.store_logits()

S=Student.Student()
S.calculate_features()
delinquent_bests = []
student_bests = []
for i in range(repeats):
    hs = S.train_student(c=0.0,T=1000.0)
    hd = S.train_delinquent()
    s = max(hs)
    d = max(hd.history["val_acc"])
    student_bests.append(s)
    delinquent_bests.append(d)

print("Best student accuracy {},\nbest delinquent accuracy {})".format(student_bests,delinquent_bests))
print("Teacher accuracy {}".format(teacher_acc))
