import Teacher
import Student
from Teacher import build_database

repeats = 5

build_database()
T = Teacher.Teacher()
T.calculate_features()
hist1 = T.train_teacher()
teacher_acc = hist1.history["val_acc"][-1]
T.store_logits()

S=Student.Student()
S.calculate_features()
delinquent_bests = []
student_bests = []
for i in range(repeats):
    s = S.train_student(1.0,0.0)
    h = S.train_delinquent()
    d = max(h.history["val_acc"])
    student_bests.append(s)
    delinquent_bests.append(d)

print("Best student accuracy {},\nbest delinquent accuracy {})".format(student_bests,delinquent_bests))
print("Teacher accuracy {}".format(teacher_acc))
