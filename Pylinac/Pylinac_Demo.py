from pylinac import CatPhan504

f = "Pylinac\Images"

ct504 = CatPhan504(f)

ct504.analyze()
print(ct504.results())

ct504.save_analyzed_image('ct504_analyzed.png')

ct504.save_analyzed_subimage('ct504_rmtf', 'rmtf')
ct504.save_analyzed_subimage('ct504_hu', 'hu')
ct504.save_analyzed_subimage('ct504_lin', 'lin')

ct504.plot_analyzed_subimage(subimage='mtf')

