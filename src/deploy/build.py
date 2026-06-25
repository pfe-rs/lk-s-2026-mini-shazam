"""
PyInstaller spec ce da bundle-uje FFmpeg binarni fajl iz tools/ kao data fajl
preko datas parametra. Prilikom build-a, spec kopira tools/ffmpeg (Linux) ili
tools/ffmpeg.exe (Windows) u _MEIPASS bundle directory.
FFmpegNormalizer.__init__ vec proverava binarni fajl relativno u odnosu na
package root, sto se u PyInstaller bundle-u razresava na sys._MEIPASS.
Nisu potrebne izmene u FFmpegNormalizer-u za PyInstaller kompatibilnost,
samo spec fajl.

ZASAD JE OVO SAMO KONCEPT. Zeleo bih da na kraju mozemo da napravimo
single-binary build sa PyInstaller-om, ali za sad pokrecemo skripte direktno.
"""
