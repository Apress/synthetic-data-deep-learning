from faker import Faker
import pandas as pnd
import numpy as np

fake_data=Faker()

fake_data=Faker("tr_TR")
print (fake_data.name())
print (fake_data.job())
print (fake_data.address())
print (fake_data.phone_number())
print (fake_data.email())

fake_data=Faker("tr_TR", use_weighting=True)
for i in range(10):
    print (fake_data.name())

print ("#############")
print (fake_data.date_of_birth())
print (fake_data.date())

print (fake_data.hostname())
print (fake_data.ipv4())


from faker import Faker
import pandas as pnd
import numpy as np

fake_data=Faker(["en_US","fr_FR","tr_TR"], use_weighting=True)
symptoms_list=["anxiety","depression","back_pain","diarrhea","fever","dizzy","cough","apnea"]
patients={}
for k in range(0,1000):
    patients[k]={}
    patients[k]['id']=k+1

    patients[k]['name']=fake_data.name()
    patients[k]['address']=fake_data.address()
    patients[k]['phone_number']=fake_data.phone_number()
    patients[k]['Date of Birth']=fake_data.date()
    patients[k]['symptoms']=np.random.choice(symptoms_list)
data_frame=pnd.DataFrame(patients).T
print (data_frame)
data_frame.to_csv("Patinets_data.csv", index=False)
    





