import pandas as pd
import numpy as np

# test_id = pd.read_csv('test_sub_ids.csv')
# train_id = pd.read_csv('train_sub_ids.csv')
# val_id = pd.read_csv('val_sub_ids.csv')
# labels = pd.read_csv('report-labels.csv')
# cxrmap = pd.read_csv('new_map.csv')
# report = pd.read_csv('reports-field-findings.tsv', sep='\t')
# view = pd.read_csv('meta.csv')

# cxrmap = cxrmap.loc[cxrmap['dicom_is_available'],:]

# cxrmap = cxrmap.merge(view, left_on='dicom_id', right_on='dicom_id')


# test = test_id.merge(cxrmap, left_on='sub_id', right_on='subject_id')
# train = train_id.merge(cxrmap, left_on='sub_id', right_on='subject_id')
# val = val_id.merge(cxrmap, left_on='sub_id', right_on='subject_id')

# test = test[['subject_id','rad_id','dicom_id','dicom_is_available','view_position']]
# train = train[['subject_id','rad_id','dicom_id','dicom_is_available','view_position']]
# val = val[['subject_id','rad_id','dicom_id','dicom_is_available','view_position']]

# report_label = report.merge(labels, left_on='rad_id', right_on='rad_id')
# test = test.merge(report_label, left_on='rad_id', right_on='rad_id')
# train = train.merge(report_label, left_on='rad_id', right_on='rad_id')
# val = val.merge(report_label, left_on='rad_id', right_on='rad_id')

# test.to_csv('test.tsv', sep='\t',index=False)
# train.to_csv('train.tsv', sep='\t',index=False)
# val.to_csv('val.tsv', sep='\t',index=False)

test = pd.read_csv('test.tsv', sep='\t')
train = pd.read_csv('train.tsv', sep='\t')
val = pd.read_csv('val.tsv', sep='\t')

test = test.loc[(test['view_position']!='LL')&(test['view_position']!='LATERAL'),:]
train = train.loc[(train['view_position']!='LL')&(train['view_position']!='LATERAL'),:]
val = val.loc[(val['view_position']!='LL')&(val['view_position']!='LATERAL'),:]

test.to_csv('test-ap.tsv', sep='\t',index=False)
train.to_csv('train-ap.tsv', sep='\t',index=False)
val.to_csv('val-ap.tsv', sep='\t',index=False)