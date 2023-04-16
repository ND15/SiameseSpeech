import musdb

mus = musdb.DB(download=True)
print(mus[0].audio)
