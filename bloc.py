import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'attendanceSystem.settings')
django.setup()

from cctv.models import Student, Attendance
import pickle
import base64
from datetime import date


# import createDataset


# this function is to add students call this function when admin adds new student
# def add_student(name, roll):
#     stud1 = Student(name=name, roll_no=roll, face_features=base64.b64encode(pickle.dumps('')), key=0, div = '')
#     stud1.save()
#     #
#     # Attribute.objects.create(slug='color', datatype=Attribute.TYPE_TEXT, name='roseaa')
#     # Attendance.objects.create(name='rose', eav__color='red')
#

# this function is to add/update face_features of specific student
def add_feature(roll, feature):
    np_bytes = pickle.dumps(feature)
    np_base64 = base64.b64encode(np_bytes)
    stud = Student.objects.get(roll_no__exact=roll)
    print(stud.face_feature)
    stud.face_feature = np_base64
    stud.save(update_fields=['face_feature'])


# function to get features of all students for classification
def get_all_features():
    stud = Student.objects.values('face_feature')
    # print(stud)
    stud_list = list(stud)
    stud_list = [d['face_feature'] for d in stud_list]
    # print(stud_list)
    feature_list = []
    for i in range(len(stud_list)):
        np_bytes = base64.b64decode(stud_list[i])
        feature_list.append(pickle.loads(np_bytes))
    # print(feature_list[0])
    return feature_list


# def compare():
#     createDataset.compare()


def get_roll_from_feature(f):
    np_bytes = pickle.dumps(f)
    np_base64 = base64.b64encode(np_bytes)
    stud = Student.objects.get(face_feature__exact=np_base64)
    return stud.name


def update_attendance():
    stud = Student.objects.filter(key=True)
    print(stud[0].name)
    for s in stud:
        roll_no = s.roll_no
        roll_no1 = '_' + str(roll_no)
        print(roll_no)
        att = Attendance.objects.get(date__exact=date.today())
        setattr(att, roll_no1, True)
        att.save(update_fields=[roll_no1])
        # att = Attendance.objects.get(date__exact=date.today())
        # print(att[roll_no1])


def changekey(naam):
    stud = Student.objects.get(name__exact=naam)
    stud.key = True
    stud.save(update_fields=['key'])


def get_all_students():
    all_studs = Student.objects.values()
    print(all_studs)
    return all_studs


def get_present_students(date=date.today()):
    present_stud = Attendance.objects.get(date__exact=date)
    present_stud = present_stud.__dict__
    present_stud = [k for k, v in present_stud.items() if v == True]
    present_stud = [s.split('_')[1] for s in present_stud]
    total = []
    for i in present_stud:
        stud = Student.objects.get(roll_no__exact=i)
        stud = stud.__dict__
        total.append(stud)
    print(total)
    return total


def set_date():
    try:
        att = Attendance(date=date.today())
        att.save()
    except Exception as e:
        pass


if __name__ == '__main__':
    get_present_students()
