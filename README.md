# DistillingNeuralNets
This repository implements the technique of distillation to transfer knowledge from a large neural net to a small one.
Not all of the functionality has been implemented yet, but the usage is as follows:
```
python Teacher.py -d #Calculate a database of the image paths and labels
python Teacher.py -f #Calculate teacher model's convolution features for each sample
python Teacher.py -t #Train the teacher model using transfer learning
python Teacher.py -l #Calculate and store the teacher's logits (rich labels)
python Student.py -f #Calculate the student model's convolution features
python Student.py -s #Train the student model using the teacher's logits
python Student.py -d #Train a delinquent model to benchmark the performance of the student
```
These commands can be combined, for example
```
python Teacher.py -d -f -t -l
```
When the project is complete, there will be one file Distill.py, which takes image labels as arguments, and returns a small keras model model "student.h5" that classifies images as one of the given labels.
