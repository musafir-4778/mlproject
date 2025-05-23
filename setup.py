from setuptools import find_packages,setup
from typing import List

HYPEN_E_DOT='-e .'
def get_requirements(file_path:str)->List[str]:
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n"," ") for req in requirements]

        if "-e." in requirements:
            requirements.remove(HYPEN_E_DOT)
    return requirements



setup(
name='mlproject',
version='0.0.1',
author='Lakshya',
author_email='2022lakshyayadav@gmail.com',
packages=find_packages(),
insall_requires=get_requirements('requirement.txt')

)