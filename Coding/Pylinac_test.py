from pylinac import CatPhan600

f = "D:\Github\MPH3013-Special-Project\Coding\mdh_images"

ct600 = CatPhan600(f)

ct600.analyze()
print(ct600.results())

ct600.save_analyzed_image('ct600_analyzed.png')

ct600.save_analyzed_subimage('ct600_rmtf', 'rmtf')
ct600.save_analyzed_subimage('ct600_hu', 'hu')
ct600.save_analyzed_subimage('ct600_lin', 'lin')

ct600.plot_analyzed_subimage(subimage='mtf')
