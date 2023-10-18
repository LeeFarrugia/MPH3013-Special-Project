from pylinac import CatPhan504

f = "/home/leefarrugia/Documents/GitHub/MPH3013-Special-Project/Images"

ct504 = CatPhan504(f)

ct504.analyze()
print(ct504.results())