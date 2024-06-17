#!/usr/bin/env python2
#encoding: UTF-8
import json
import sys;sys.path.append('./')
import zipfile
import re
import sys
import os
import numpy as np
import codecs
import functools
import importlib
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

from datetime import datetime
import shutil

def print_help():
    sys.stdout.write('Usage: python %s.py -g=<gtFile> -s=<submFile> [-o=<outputFolder> -p=<jsonParams>]' %sys.argv[0])
    sys.exit(2)
    

def load_zip_file_keys(file,fileNameRegExp=''):
    """
    Returns an array with the entries of the ZIP file that match with the regular expression.
    The key's are the names or the file or the capturing group definied in the fileNameRegExp
    """
    try:
        zip_root = os.path.dirname(file)
        zip_name = os.path.basename(file)
        # time_id = datetime.now().strftime(r'%m%d_%H%M%S')
        time_id = datetime.now().strftime(r'%m%d_%H%M_%S%f')
        tmp_zip_name = f"{os.path.splitext(zip_name)[0]}_{time_id}{os.path.splitext(zip_name)[-1]}"
        tmp_zip_name = os.path.join(zip_root, tmp_zip_name)
        shutil.copyfile(file, tmp_zip_name)
        archive=zipfile.ZipFile(tmp_zip_name, mode='r', allowZip64=True)
        os.remove(tmp_zip_name)
        # archive=zipfile.ZipFile(file, mode='r', allowZip64=True)
    except :
        raise Exception('Error loading the ZIP archive.')

    pairs = []
    
    for name in archive.namelist():
        addFile = True
        keyName = name
        if fileNameRegExp!="":
            m = re.match(fileNameRegExp,name)
            if m == None:
                addFile = False
            else:
                if len(m.groups())>0:
                    keyName = m.group(1)
                    
        if addFile:
            pairs.append( keyName )
                
    return pairs
    

def load_zip_file(file,fileNameRegExp='',allEntries=False):
    """
    Returns an array with the contents (filtered by fileNameRegExp) of a ZIP file.
    The key's are the names or the file or the capturing group definied in the fileNameRegExp
    allEntries validates that all entries in the ZIP file pass the fileNameRegExp
    """
    try:
        zip_root = os.path.dirname(file)
        zip_name = os.path.basename(file)
        # time_id = datetime.now().strftime(r'%m%d_%H%M%S')
        time_id = datetime.now().strftime(r'%m%d_%H%M_%S%f')
        tmp_zip_name = f"{os.path.splitext(zip_name)[0]}_{time_id}{os.path.splitext(zip_name)[-1]}"
        tmp_zip_name = os.path.join(zip_root, tmp_zip_name)
        shutil.copyfile(file, tmp_zip_name)
        archive=zipfile.ZipFile(tmp_zip_name, mode='r', allowZip64=True)
        os.remove(tmp_zip_name)
        # archive=zipfile.ZipFile(file, mode='r', allowZip64=True)
    except :
        raise Exception('Error loading the ZIP archive')    

    pairs = []
    for name in archive.namelist():
        addFile = True
        keyName = name
        if fileNameRegExp!="":
            m = re.match(fileNameRegExp,name)
            if m == None:
                addFile = False
            else:
                if len(m.groups())>0:
                    keyName = m.group(1)
        
        if addFile:
            pairs.append( [ keyName , archive.read(name)] )
        else:
            if allEntries:
                raise Exception('ZIP entry not valid: %s' %name)             

    return dict(pairs)
	
def decode_utf8(raw):
    """
    Returns a Unicode object on success, or None on failure
    """
    try:
        raw = codecs.decode(raw,'utf-8', 'replace')
        #extracts BOM if exists
        raw = raw.encode('utf8')
        if raw.startswith(codecs.BOM_UTF8):
            raw = raw.replace(codecs.BOM_UTF8, '', 1)
        return raw.decode('utf-8')
    except:
       return None
   
def validate_lines_in_file(fileName,file_contents,CRLF=True,LTRB=True,withTranscription=False,withConfidence=False,imWidth=0,imHeight=0):
    """
    This function validates that all lines of the file calling the Line validation function for each line
    """
    utf8File = decode_utf8(file_contents)
    if (utf8File is None) :
        raise Exception("The file %s is not UTF-8" %fileName)

    lines = utf8File.split( "\r\n" if CRLF else "\n" )
    for line in lines:
        line = line.replace("\r","").replace("\n","")
        if(line != ""):
            try:
                validate_tl_line(line,LTRB,withTranscription,withConfidence,imWidth,imHeight)
            except Exception as e:
                raise Exception(("Line in sample not valid. Sample: %s Line: %s Error: %s" %(fileName,line,str(e))).encode('utf-8', 'replace'))
    
   
   
def validate_tl_line(line,LTRB=True,withTranscription=True,withConfidence=True,imWidth=0,imHeight=0):
    """
    Validate the format of the line. If the line is not valid an exception will be raised.
    If maxWidth and maxHeight are specified, all points must be inside the imgage bounds.
    Posible values are:
    LTRB=True: xmin,ymin,xmax,ymax[,confidence][,transcription] 
    LTRB=False: x1,y1,x2,y2,x3,y3,x4,y4[,confidence][,transcription] 
    """
    get_tl_line_values(line,LTRB,withTranscription,withConfidence,imWidth,imHeight)
    
   
def get_tl_line_values(line,LTRB=True,withTranscription=False,withConfidence=False,imWidth=0,imHeight=0):
    """
    Validate the format of the line. If the line is not valid an exception will be raised.
    If maxWidth and maxHeight are specified, all points must be inside the imgage bounds.
    Posible values are:
    LTRB=True: xmin,ymin,xmax,ymax[,confidence][,transcription] 
    LTRB=False: x1,y1,x2,y2,x3,y3,x4,y4[,confidence][,transcription] 
    Returns values from a textline. Points , [Confidences], [Transcriptions]
    """
    confidence = 0.0
    transcription = "";
    points = []
    
    numPoints = 4;
    
    if LTRB:
    
        numPoints = 4;
        
        if withTranscription and withConfidence:
            m = re.match(r'^\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*([0-9]+)\s*,\s*([0-9]+)\s*,\s*([0-1].?[0-9]*)\s*,(.*)$',line)
            if m == None :
                m = re.match(r'^\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*([0-9]+)\s*,\s*([0-9]+)\s*,\s*([0-1].?[0-9]*)\s*,(.*)$',line)
                raise Exception("Format incorrect. Should be: xmin,ymin,xmax,ymax,confidence,transcription")
        elif withConfidence:
            m = re.match(r'^\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*([0-9]+)\s*,\s*([0-9]+)\s*,\s*([0-1].?[0-9]*)\s*$',line)
            if m == None :
                raise Exception("Format incorrect. Should be: xmin,ymin,xmax,ymax,confidence")
        elif withTranscription:
            m = re.match(r'^\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*([0-9]+)\s*,\s*([0-9]+)\s*,(.*)$',line)
            if m == None :
                raise Exception("Format incorrect. Should be: xmin,ymin,xmax,ymax,transcription")
        else:
            m = re.match(r'^\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*([0-9]+)\s*,\s*([0-9]+)\s*,?\s*$',line)
            if m == None :
                raise Exception("Format incorrect. Should be: xmin,ymin,xmax,ymax")
            
        xmin = int(m.group(1))
        ymin = int(m.group(2))
        xmax = int(m.group(3))
        ymax = int(m.group(4))
        if(xmax<xmin):
                raise Exception("Xmax value (%s) not valid (Xmax < Xmin)." %(xmax))
        if(ymax<ymin):
                raise Exception("Ymax value (%s)  not valid (Ymax < Ymin)." %(ymax))  

        points = [ float(m.group(i)) for i in range(1, (numPoints+1) ) ]
        
        if (imWidth>0 and imHeight>0):
            validate_point_inside_bounds(xmin,ymin,imWidth,imHeight);
            validate_point_inside_bounds(xmax,ymax,imWidth,imHeight);

    else:
        
        numPoints = 8;
        
        if withTranscription and withConfidence:
            m = re.match(r'^\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*([0-1].?[0-9]*)\s*,(.*)$',line)
            if m == None :
                raise Exception("Format incorrect. Should be: x1,y1,x2,y2,x3,y3,x4,y4,confidence,transcription")
        elif withConfidence:
            m = re.match(r'^\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*([0-1].?[0-9]*)\s*$',line)
            if m == None :
                raise Exception("Format incorrect. Should be: x1,y1,x2,y2,x3,y3,x4,y4,confidence")
        elif withTranscription:
            m = re.match(r'^\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,(.*)$',line)
            if m == None :
                raise Exception("Format incorrect. Should be: x1,y1,x2,y2,x3,y3,x4,y4,transcription")
        else:
            m = re.match(r'^\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*$',line)
            if m == None :
                raise Exception("Format incorrect. Should be: x1,y1,x2,y2,x3,y3,x4,y4")
            
        points = [ float(m.group(i)) for i in range(1, (numPoints+1) ) ]

        points = validate_clockwise_points(points)
        
        if (imWidth>0 and imHeight>0):
            validate_point_inside_bounds(points[0],points[1],imWidth,imHeight);
            validate_point_inside_bounds(points[2],points[3],imWidth,imHeight);
            validate_point_inside_bounds(points[4],points[5],imWidth,imHeight);
            validate_point_inside_bounds(points[6],points[7],imWidth,imHeight);
            
    
    if withConfidence:
        try:
            confidence = float(m.group(numPoints+1))
        except ValueError:
            raise Exception("Confidence value must be a float")       
            
    if withTranscription:
        posTranscription = numPoints + (2 if withConfidence else 1)
        transcription = m.group(posTranscription)
        m2 = re.match(r'^\s*\"(.*)\"\s*$',transcription)
        if m2 != None : #Transcription with double quotes, we extract the value and replace escaped characters
            transcription = m2.group(1).replace("\\\\", "\\").replace("\\\"", "\"")
    
    return points,confidence,transcription
    
            
def validate_point_inside_bounds(x,y,imWidth,imHeight):
    if(x<0 or x>imWidth):
            raise Exception("X value (%s) not valid. Image dimensions: (%s,%s)" %(x,imWidth,imHeight))
    if(y<0 or y>imHeight):
            raise Exception("Y value (%s)  not valid. Image dimensions: (%s,%s) Sample: %s Line:%s" %(y,imWidth,imHeight))


def sort_points(points):
    """Sort arbitory points in clockwise order. Reference:
    https://stackoverflow.com/a/6989383.
    Args:
        points (list[ndarray] or ndarray or list[list]): A list of unsorted
            boundary points.
    Returns:
        list[ndarray]: A list of points sorted in clockwise order.
    """


    points = np.array(points)
    center = np.mean(points, axis=0)

    def cmp(a, b):
        oa = a - center
        ob = b - center

        # Some corner cases
        if oa[0] >= 0 and ob[0] < 0:
            return 1
        if oa[0] < 0 and ob[0] >= 0:
            return -1

        prod = np.cross(oa, ob)
        if prod > 0:
            return 1
        if prod < 0:
            return -1

        # a, b are on the same line from the center
        return 1 if (oa**2).sum() < (ob**2).sum() else -1

    return sorted(points, key=functools.cmp_to_key(cmp))

def validate_clockwise_points(points):
    """
    Validates that the points that the 4 points that dlimite a polygon are in clockwise order.
    """
    
    if len(points) != 8:
        raise Exception("Points list not valid." + str(len(points)))
    
    point = [
                [int(points[0]) , int(points[1])],
                [int(points[2]) , int(points[3])],
                [int(points[4]) , int(points[5])],
                [int(points[6]) , int(points[7])]
            ]
    edge = [
                ( point[1][0] - point[0][0])*( point[1][1] + point[0][1]),
                ( point[2][0] - point[1][0])*( point[2][1] + point[1][1]),
                ( point[3][0] - point[2][0])*( point[3][1] + point[2][1]),
                ( point[0][0] - point[3][0])*( point[0][1] + point[3][1])
    ]
    
    summatory = edge[0] + edge[1] + edge[2] + edge[3];
    if summatory>0:
        sorted_points = sort_points(point)
        sorted_points = np.array(sorted_points).flatten().tolist()
        # [narray(x1,y1),....]
        # sorted_points = [x for x in ]
        # import pdb;pdb.set_trace()
        # raise Exception("Points are not clockwise. The coordinates of bounding quadrilaterals have to be given in clockwise order. Regarding the correct interpretation of 'clockwise' remember that the image coordinate system used is the standard one, with the image origin at the upper left, the X axis extending to the right and Y axis extending downwards.")
        return sorted_points
    else:
        return points

def get_tl_line_values_from_file_contents(content,CRLF=True,LTRB=True,withTranscription=False,withConfidence=False,imWidth=0,imHeight=0,sort_by_confidences=True):
    """
    Returns all points, confindences and transcriptions of a file in lists. Valid line formats:
    xmin,ymin,xmax,ymax,[confidence],[transcription]
    x1,y1,x2,y2,x3,y3,x4,y4,[confidence],[transcription]
    """
    pointsList = []
    transcriptionsList = []
    confidencesList = []
    
    lines = content.split( "\r\n" if CRLF else "\n" )
    for line in lines:
        line = line.replace("\r","").replace("\n","")
        if(line != "") :
            points, confidence, transcription = get_tl_line_values(line,LTRB,withTranscription,withConfidence,imWidth,imHeight);
            pointsList.append(points)
            transcriptionsList.append(transcription)
            confidencesList.append(confidence)

    if withConfidence and len(confidencesList)>0 and sort_by_confidences:
        import numpy as np
        sorted_ind = np.argsort(-np.array(confidencesList))
        confidencesList = [confidencesList[i] for i in sorted_ind]
        pointsList = [pointsList[i] for i in sorted_ind]
        transcriptionsList = [transcriptionsList[i] for i in sorted_ind]        
        
    return pointsList,confidencesList,transcriptionsList

def main_evaluation(p,default_evaluation_params_fn,validate_data_fn,evaluate_method_fn,show_result=True,per_sample=True):
    """
    This process validates a method, evaluates it and if it succed generates a ZIP file with a JSON entry for each sample.
    Params:
    p: Dictionary of parmeters with the GT/submission locations. If None is passed, the parameters send by the system are used.
    default_evaluation_params_fn: points to a function that returns a dictionary with the default parameters used for the evaluation
    validate_data_fn: points to a method that validates the corrct format of the submission
    evaluate_method_fn: points to a function that evaluated the submission and return a Dictionary with the results
    """
    
    if (p == None):
        p = dict([s[1:].split('=') for s in sys.argv[1:]])
        if(len(sys.argv)<3):
            print_help()

    evalParams = default_evaluation_params_fn()
    if 'p' in p.keys():
        evalParams.update( p['p'] if isinstance(p['p'], dict) else json.loads(p['p'][1:-1]) )

    resDict={'calculated':True,'Message':'','method':'{}','per_sample':'{}'}    
    try:
        validate_data_fn(p['g'], p['s'], evalParams)  
        evalData = evaluate_method_fn(p['g'], p['s'], evalParams)
        resDict.update(evalData)
        
    except Exception as e:
        resDict['Message']= str(e)
        resDict['calculated']=False

    if 'o' in p:
        if not os.path.exists(p['o']):
            os.makedirs(p['o'])

        time_id = datetime.now().strftime(r'%m%d_%H%M_%S%f')
        resultsOutputname = p['o'] + f'/results_{time_id}.zip'
        # resultsOutputname = p['o'] + '/results.zip'
        outZip = zipfile.ZipFile(resultsOutputname, mode='w', allowZip64=True)

        del resDict['per_sample']
        if 'output_items' in resDict.keys():
            del resDict['output_items']

        outZip.writestr('method.json',json.dumps(resDict))
        
    if not resDict['calculated']:
        if show_result:
            sys.stderr.write('Error!\n'+ resDict['Message']+'\n\n')
        if 'o' in p:
            outZip.close()
        return resDict
    
    if 'o' in p:
        if per_sample == True:
            for k,v in evalData['per_sample'].items():
                outZip.writestr( k + '.json',json.dumps(v)) 

            if 'output_items' in evalData.keys():
                for k, v in evalData['output_items'].items():
                    outZip.writestr( k,v) 

        outZip.close()

    # if show_result:
    #     sys.stdout.write("Calculated!")
    #     sys.stdout.write("\n")
    #     sys.stdout.write(json.dumps(resDict['e2e_method']))
    #     sys.stdout.write("\n")
    
    return resDict


def main_validation(default_evaluation_params_fn,validate_data_fn):
    """
    This process validates a method
    Params:
    default_evaluation_params_fn: points to a function that returns a dictionary with the default parameters used for the evaluation
    validate_data_fn: points to a method that validates the corrct format of the submission
    """    
    try:
        p = dict([s[1:].split('=') for s in sys.argv[1:]])
        evalParams = default_evaluation_params_fn()
        if 'p' in p.keys():
            evalParams.update( p['p'] if isinstance(p['p'], dict) else json.loads(p['p'][1:-1]) )

        validate_data_fn(p['g'], p['s'], evalParams)              
        print('SUCCESS')
        sys.exit(0)
    except Exception as e:
        print(str(e))
        sys.exit(101)
