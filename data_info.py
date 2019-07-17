
class timit_info(object):
	wavefolder = '/Volumes/WDBlueTM/timit_feature/timit_wav'
	outfolder = '/Volumes/WDBlueTM/timit_feature/timit_onefile/25ms_10ms'
	extension='.wav'
	idctcoefffile=''

	def generatefunc(self,wavefilepath):
		personname = wavefilepath.split('/')[-2]
		filename = wavefilepath.split('/')[-1].split('.')[0]
		return personname, filename


class librispeech_info(object):
	wavefolder='/diskc/datafile/thch30/LibriSpeech/wav'
	outfolder='/diskc/datafile/thch30/LibriSpeech/onefile'
	idctcoefffile = ''
	extension = '.wav'


	def generatefunc(self,wavefilepath):
		f = wavefilepath.split('/')[-1]
		personname = f.split('-')[0]
		filename = f.split('-')[1] + '_' + f.split('-')[2]
		return personname, filename