# +
import pandas as pd
import numpy as np
import re

class ICDCodesGrouper(object):
    
    def __init__(self,ccs_path,icd9_chapter_path):
        self.ccs = self.CCSSingleDiagnosis(ccs_path)
        self.icd9chapters = self.ICD9_CM_Chapters(icd9_chapter_path)
    
    class CCSSingleDiagnosis:
        def __init__(self,file = None):

            if file is None:
                file = 'CCS-SingleDiagnosisGrouper.txt'
            file = open(file,"r")
            content = file.read()
            file.close()
            lookup = {}
            groups = re.findall('(\d+\s+[A-Z].*(\n.+)+)',content)
            for group in groups:
                parsed = group[0].split()
                for code in parsed[2:]:
                    lookup[code] = int(parsed[0])
            self.__lookup = lookup

        def lookup(self,code):
            """
            Given an icd9 code, returns the corresponding ccs code.
            
            code can be either a string or a pd.Series
            
            Returns:
                -1: code received isn't a string
                0: code doens't exist
                >0: corresponding ccs code
            """
            
            def lookup_single(code : str):
                
                try:
                    if type(code) == str:
                        return self.__lookup[code]
                    else:
                        return -1
                except:
                    return 0
            
            if type(code) == pd.Series:
                return code.apply(lookup_single)
            else:
                return lookup_single(code)
        
    class ICD9_CM_Chapters:
        def __init__(self,filepath):
            # creates self.chapters_num & self.chapters_char & self.bins
            self.__preprocess_chapters(filepath)

        def lookup(self,code):
            """
            code can be either a string or a pd.Series

            Returns Chapter
            """

            def single_lookup(code_):

                def char_lookup(code_):
                    """
                    When the code starts by a char, it's either E or V
                    """
                    if code_[0] == 'E':
                        return 19
                    elif code_[0] == 'V':
                        return 18
                    return 0

                def int_lookup(code_):
                    level_3_code = int(code_[:3])
                    pos = np.digitize(level_3_code,self.bins)
                    chapter = self.chapters_num.Chapter.iloc[pos-1]
                    return chapter

                if code_[0] in ['E','V']:
                    return char_lookup(code_)
                else:
                    return int_lookup(code_)

            def batch_lookup(codes : pd.Series):
                
                
                mask_is_na = codes.isna()
                codes_na = codes.loc[mask_is_na]
                codes_not_na = codes.loc[~mask_is_na]
                
                
                mask_is_alpha = codes_not_na.apply(lambda x: (x[0] == 'E') | (x[0] == 'V'))
                codes_char = (codes_not_na
                              .loc[mask_is_alpha]
                              .copy()
                              .apply(lambda x:x[0]) # only need first character to identify chapter
                             )
                
                codes_num = (codes_not_na
                             .loc[~codes_not_na.index.isin(codes_char.index)]
                             .copy()
                             .apply(lambda x: x[:3]) # only need first 3 characters to identify chapter
                             .astype(int)
                            )
                
                
                # get chapters of numeric codes
                num_chapters = (pd.Series(data=np.digitize(codes_num,self.bins),
                                          index=codes_num.index)
                               )
                
                
                char_chapters = codes_char.apply(single_lookup)
                
                na_chapters = pd.Series(np.nan,index=codes_na.index)
                result = (pd.concat([num_chapters,char_chapters,na_chapters],axis='rows') # merge chapters of numerical & alpha codes
                          .loc[codes.index] # get original order
                         )
                return result
            
            if type(code) not in [str,pd.Series]:
                return -1

            if type(code) == str:
                return single_lookup(code)
            else:
                return batch_lookup(code)

        def __preprocess_chapters(self,filepath):
            
            self.chapters = pd.read_csv(filepath)

            # preprocess chapters dataframe: split into alpha vs numeric
            self.chapters['start_range'] = self.chapters['Code Range'].apply(lambda x: x[:x.find('-')])

            chapters_char = (self.chapters
                           .loc[self.chapters.start_range
                                .apply(lambda x: len(re.findall('^(E|V)',x)) > 0),
                                ['Chapter','Description','Code Range']
                               ]
                          )
            # only need the first letter
            chapters_char.loc[:,'Code Range'] = chapters_char['Code Range'].apply(lambda x: x[0])

            chapters_num = (self.chapters
                          .loc[~self.chapters.index.isin(chapters_char.index)]
                          .astype({'start_range':int}).copy()
                         )

            bins = chapters_num.start_range.tolist()
            # need to add last interval
            bins.append(bins[-1]+205)

            self.bins = bins
            self.chapters_num = chapters_num
            self.chapters_char = chapters_char
# -

# # Testing zone

# from Mimic import Mimic
#
# mimic = Mimic()

# ## CCS

# grouper = ICDCodesGrouper('CCS-SingleDiagnosisGrouper.txt')

# diag = mimic.read_diagnoses()
# diag.head(2)
# diag.shape

# res = grouper.ccs.lookup(diag['ICD9_CODE'])
# res.head(3)
# res.shape
# print('How many problems (expecting zero, aka TRUE)')
# res[(res == 0) | (res == -1)].empty
# print('Number of unique classes')
# res.nunique()
# print('Distribution of CCS categories')
# res.value_counts()

# ## ICD9 chapters

# grouper = ICDCodesGrouper('CCS-SingleDiagnosisGrouper.txt','icd9-CM-code-chapter-en=PT.csv')

# diag = mimic.read_diagnoses()
# diag.head(2)
# diag.shape

# # %%time
# diag.assign(chapter=lambda row: grouper.icd9chapters.lookup(row.ICD9_CODE))

#
