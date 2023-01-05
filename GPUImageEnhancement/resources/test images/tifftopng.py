from wand.image import Image 

with Image(filename ='jetplane.tif') as Sampleimg:  
    Sampleimg.format = 'png' 
    Sampleimg.save(filename ='jetplane.png')