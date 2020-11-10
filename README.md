# attack demo


To generate attacked images, use the following example. Make sure to specify the necessary commands. 
```bash
python main.py --area C --attack_type PGD_l2 --attack_goal N --model_type resnet18 --strategy WB 
```
And then the following commands will be executed.
```
python PGD.py --area C --attack_type PGD_l2 --attack_goal N --model_type resnet18 --strategy WB
```
The arguments behind are the ones in PGD.py file, which are be used to set the conresponding parameters.

To submit your attacks, name your main attack file with the attack name such as 'PGD.py'. Originize your source files and make sure the attack can run such as the following example.
```bash
python PGD.py --area C --attack_type PGD_l2 --attack_goal N --model_type resnet18 --strategy WB
```
**How to add your own attacks?**   
In main.py, make sure to keep ***attack_type*** same with ***attack_type***  in your own attack files. For example,  'PGD_linf' and 'PGD_l2' are the attack_types used in PGD.py. So the codes in main.py shoule be like:
```bash
if attack_type == 'PGD_linf' or 'PGD_l2':
    attack = 'PGD'
```
Here ***attack*** is the name of your attack file.  
Another example is adversarial patch attack.
```bash
if attack_type == 'Patch':
    attack = 'Patch_Attack'
```
The codebook is here:  
## Codebook:  
<body lang="ZH-CN" style="tab-interval:21.0pt;text-justify-trim:punctuation">

<div class="WordSection1" style="layout-grid:15.6pt">

<table class="MsoNormalTable" border="0" cellspacing="0" cellpadding="0" width="670" style="width:502.85pt;border-collapse:collapse;mso-yfti-tbllook:1184;
 mso-padding-alt:0cm 0cm 0cm 0cm">
 <tbody><tr style="mso-yfti-irow:0;mso-yfti-firstrow:yes;height:15.75pt">
  <td width="264" colspan="2" style="width:198.05pt;border:solid #BFBFBF 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:15.75pt">
  <p class="MsoNormal" align="center" style="text-align:center;mso-pagination:widow-orphan"><span lang="EN-US" style="font-size:10.0pt;font-family:&quot;Arial&quot;,sans-serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt">head</span><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-ascii-font-family:等线;mso-fareast-font-family:等线;mso-hansi-font-family:
  等线;mso-bidi-font-family:&quot;Times New Roman&quot;;mso-font-kerning:0pt"><o:p></o:p></span></p>
  </td>
  <td width="406" colspan="3" style="width:304.8pt;border:solid #BFBFBF 1.0pt;
  border-left:none;padding:0cm 5.4pt 0cm 5.4pt;height:15.75pt">
  <p class="MsoNormal" align="center" style="text-align:center;mso-pagination:widow-orphan"><span lang="EN-US" style="font-size:10.0pt;font-family:&quot;Arial&quot;,sans-serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt">details</span><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-ascii-font-family:等线;mso-fareast-font-family:等线;mso-hansi-font-family:
  等线;mso-bidi-font-family:&quot;Times New Roman&quot;;mso-font-kerning:0pt"><o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:1;height:15.75pt">
  <td width="141" rowspan="2" style="width:105.95pt;border:solid #BFBFBF 1.0pt;
  border-top:none;padding:0cm 5.4pt 0cm 5.4pt;height:15.75pt">
  <p class="MsoNormal" align="center" style="text-align:center;mso-pagination:widow-orphan"><span lang="EN-US" style="font-size:10.0pt;font-family:&quot;Arial&quot;,sans-serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt">area</span><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-ascii-font-family:等线;mso-fareast-font-family:等线;mso-hansi-font-family:
  等线;mso-bidi-font-family:&quot;Times New Roman&quot;;mso-font-kerning:0pt"><o:p></o:p></span></p>
  </td>
  <td width="123" rowspan="2" style="width:92.1pt;border-top:none;border-left:none;
  border-bottom:solid #BFBFBF 1.0pt;border-right:solid #BFBFBF 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:15.75pt">
  <p class="MsoNormal" align="center" style="text-align:center;mso-pagination:widow-orphan"><span lang="EN-US" style="font-size:10.0pt;font-family:&quot;Arial&quot;,sans-serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt">attack type</span><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-ascii-font-family:等线;mso-fareast-font-family:等线;mso-hansi-font-family:
  等线;mso-bidi-font-family:&quot;Times New Roman&quot;;mso-font-kerning:0pt"><o:p></o:p></span></p>
  </td>
  <td width="76" rowspan="2" style="width:2.0cm;border-top:none;border-left:none;
  border-bottom:solid #BFBFBF 1.0pt;border-right:solid #BFBFBF 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:15.75pt">
  <p class="MsoNormal" align="center" style="text-align:center;mso-pagination:widow-orphan"><span lang="EN-US" style="font-size:10.0pt;font-family:&quot;Arial&quot;,sans-serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt">attack goal</span><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-ascii-font-family:等线;mso-fareast-font-family:等线;mso-hansi-font-family:
  等线;mso-bidi-font-family:&quot;Times New Roman&quot;;mso-font-kerning:0pt"><o:p></o:p></span></p>
  </td>
  <td width="331" colspan="2" style="width:248.1pt;border-top:none;border-left:
  none;border-bottom:solid #BFBFBF 1.0pt;border-right:solid #BFBFBF 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:15.75pt">
  <p class="MsoNormal" align="center" style="text-align:center;mso-pagination:widow-orphan"><span lang="EN-US" style="font-size:10.0pt;font-family:&quot;Arial&quot;,sans-serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt">attacker knowledge</span><span lang="EN-US" style="mso-bidi-font-size:10.5pt;mso-ascii-font-family:等线;mso-fareast-font-family:
  等线;mso-hansi-font-family:等线;mso-bidi-font-family:&quot;Times New Roman&quot;;
  mso-font-kerning:0pt"><o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:2;height:5.7pt">
  <td width="170" style="width:127.6pt;border-top:none;border-left:none;
  border-bottom:solid #BFBFBF 1.0pt;border-right:solid #BFBFBF 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:5.7pt">
  <p class="MsoNormal" align="center" style="text-align:center;mso-pagination:widow-orphan"><span lang="EN-US" style="font-size:10.0pt;font-family:&quot;Arial&quot;,sans-serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt">model type</span><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-ascii-font-family:等线;mso-fareast-font-family:等线;mso-hansi-font-family:
  等线;mso-bidi-font-family:&quot;Times New Roman&quot;;mso-font-kerning:0pt"><o:p></o:p></span></p>
  </td>
  <td width="161" style="width:120.5pt;border-top:none;border-left:none;
  border-bottom:solid #BFBFBF 1.0pt;border-right:solid #BFBFBF 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:5.7pt">
  <p class="MsoNormal" align="center" style="text-align:center;mso-pagination:widow-orphan"><span lang="EN-US" style="font-size:10.0pt;font-family:&quot;Arial&quot;,sans-serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt">strategy</span><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-ascii-font-family:等线;mso-fareast-font-family:等线;mso-hansi-font-family:
  等线;mso-bidi-font-family:&quot;Times New Roman&quot;;mso-font-kerning:0pt"><o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:3;height:15.75pt">
  <td width="141" style="width:105.95pt;border:solid #BFBFBF 1.0pt;border-top:
  none;padding:0cm 5.4pt 0cm 5.4pt;height:15.75pt">
  <p class="MsoNormal" align="center" style="text-align:center;mso-pagination:widow-orphan"><span lang="EN-US" style="font-size:10.0pt;font-family:&quot;Arial&quot;,sans-serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt">C -- Classification</span><span lang="EN-US" style="mso-bidi-font-size:10.5pt;mso-ascii-font-family:等线;mso-fareast-font-family:
  等线;mso-hansi-font-family:等线;mso-bidi-font-family:&quot;Times New Roman&quot;;
  mso-font-kerning:0pt"><o:p></o:p></span></p>
  </td>
  <td width="123" style="width:92.1pt;border-top:none;border-left:none;
  border-bottom:solid #BFBFBF 1.0pt;border-right:solid #BFBFBF 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:15.75pt">
  <p class="MsoNormal" align="center" style="text-align:center;mso-pagination:widow-orphan"><span lang="EN-US" style="font-size:10.0pt;font-family:&quot;Arial&quot;,sans-serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt">0 -- non-attack</span><span lang="EN-US" style="mso-bidi-font-size:10.5pt;mso-ascii-font-family:等线;mso-fareast-font-family:
  等线;mso-hansi-font-family:等线;mso-bidi-font-family:&quot;Times New Roman&quot;;
  mso-font-kerning:0pt"><o:p></o:p></span></p>
  </td>
  <td width="76" style="width:2.0cm;border-top:none;border-left:none;border-bottom:
  solid #BFBFBF 1.0pt;border-right:solid #BFBFBF 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:15.75pt">
  <p class="MsoNormal" align="center" style="text-align:center;mso-pagination:widow-orphan"><span lang="EN-US" style="font-size:10.0pt;font-family:&quot;Arial&quot;,sans-serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt">T -- target</span><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-ascii-font-family:等线;mso-fareast-font-family:等线;mso-hansi-font-family:
  等线;mso-bidi-font-family:&quot;Times New Roman&quot;;mso-font-kerning:0pt"><o:p></o:p></span></p>
  </td>
  <td width="170" style="width:127.6pt;border-top:none;border-left:none;
  border-bottom:solid #BFBFBF 1.0pt;border-right:solid #BFBFBF 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:15.75pt">
  <p class="MsoNormal" align="center" style="text-align:center;mso-pagination:widow-orphan"><span lang="EN-US" style="font-size:10.0pt;font-family:&quot;Arial&quot;,sans-serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt">V16 -- VGG16</span><span lang="EN-US" style="mso-bidi-font-size:10.5pt;mso-ascii-font-family:等线;mso-fareast-font-family:
  等线;mso-hansi-font-family:等线;mso-bidi-font-family:&quot;Times New Roman&quot;;
  mso-font-kerning:0pt"><o:p></o:p></span></p>
  </td>
  <td width="161" style="width:120.5pt;border-top:none;border-left:none;
  border-bottom:solid #BFBFBF 1.0pt;border-right:solid #BFBFBF 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:15.75pt">
  <p class="MsoNormal" align="center" style="text-align:center;mso-pagination:widow-orphan"><span lang="EN-US" style="font-size:10.0pt;font-family:&quot;Arial&quot;,sans-serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt">WA -- White Box</span><span lang="EN-US" style="mso-bidi-font-size:10.5pt;mso-ascii-font-family:等线;mso-fareast-font-family:
  等线;mso-hansi-font-family:等线;mso-bidi-font-family:&quot;Times New Roman&quot;;
  mso-font-kerning:0pt"><o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:4;height:15.75pt">
  <td width="141" style="width:105.95pt;border:solid #BFBFBF 1.0pt;border-top:
  none;padding:0cm 5.4pt 0cm 5.4pt;height:15.75pt">
  <p class="MsoNormal" align="center" style="text-align:center;mso-pagination:widow-orphan"><span lang="EN-US" style="font-size:10.0pt;font-family:&quot;Arial&quot;,sans-serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt">D -- Detection</span><span lang="EN-US" style="mso-bidi-font-size:10.5pt;mso-ascii-font-family:等线;mso-fareast-font-family:
  等线;mso-hansi-font-family:等线;mso-bidi-font-family:&quot;Times New Roman&quot;;
  mso-font-kerning:0pt"><o:p></o:p></span></p>
  </td>
  <td width="123" style="width:92.1pt;border-top:none;border-left:none;
  border-bottom:solid #BFBFBF 1.0pt;border-right:solid #BFBFBF 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:15.75pt">
  <p class="MsoNormal" align="center" style="text-align:center;mso-pagination:widow-orphan"><span lang="EN-US" style="font-size:10.0pt;font-family:&quot;Arial&quot;,sans-serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt">1 --&nbsp;<span class="SpellE">lp</span>&nbsp;attack</span><span lang="EN-US" style="mso-bidi-font-size:10.5pt;mso-ascii-font-family:等线;
  mso-fareast-font-family:等线;mso-hansi-font-family:等线;mso-bidi-font-family:
  &quot;Times New Roman&quot;;mso-font-kerning:0pt"><o:p></o:p></span></p>
  </td>
  <td width="76" style="width:2.0cm;border-top:none;border-left:none;border-bottom:
  solid #BFBFBF 1.0pt;border-right:solid #BFBFBF 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:15.75pt">
  <p class="MsoNormal" align="center" style="text-align:center;mso-pagination:widow-orphan"><span lang="EN-US" style="font-size:10.0pt;font-family:&quot;Arial&quot;,sans-serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt">N -- non-target</span><span lang="EN-US" style="mso-bidi-font-size:10.5pt;mso-ascii-font-family:等线;mso-fareast-font-family:
  等线;mso-hansi-font-family:等线;mso-bidi-font-family:&quot;Times New Roman&quot;;
  mso-font-kerning:0pt"><o:p></o:p></span></p>
  </td>
  <td width="170" style="width:127.6pt;border-top:none;border-left:none;
  border-bottom:solid #BFBFBF 1.0pt;border-right:solid #BFBFBF 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:15.75pt">
  <p class="MsoNormal" align="center" style="text-align:center;mso-pagination:widow-orphan"><span lang="EN-US" style="font-size:10.0pt;font-family:&quot;Arial&quot;,sans-serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt">V19 -- VGG19</span><span lang="EN-US" style="mso-bidi-font-size:10.5pt;mso-ascii-font-family:等线;mso-fareast-font-family:
  等线;mso-hansi-font-family:等线;mso-bidi-font-family:&quot;Times New Roman&quot;;
  mso-font-kerning:0pt"><o:p></o:p></span></p>
  </td>
  <td width="161" style="width:120.5pt;border-top:none;border-left:none;
  border-bottom:solid #BFBFBF 1.0pt;border-right:solid #BFBFBF 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:15.75pt">
  <p class="MsoNormal" align="center" style="text-align:center;mso-pagination:widow-orphan"><span lang="EN-US" style="font-size:10.0pt;font-family:&quot;Arial&quot;,sans-serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt">TB -- Transfer based Black Box</span><span lang="EN-US" style="mso-bidi-font-size:10.5pt;mso-ascii-font-family:等线;
  mso-fareast-font-family:等线;mso-hansi-font-family:等线;mso-bidi-font-family:
  &quot;Times New Roman&quot;;mso-font-kerning:0pt"><o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:5;height:15.75pt">
  <td width="141" style="width:105.95pt;border:solid #BFBFBF 1.0pt;border-top:
  none;padding:0cm 5.4pt 0cm 5.4pt;height:15.75pt">
  <p class="MsoNormal" align="center" style="text-align:center;mso-pagination:widow-orphan"><span lang="EN-US" style="font-size:10.0pt;font-family:&quot;Arial&quot;,sans-serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt">F -- Face recognition</span><span lang="EN-US" style="mso-bidi-font-size:10.5pt;mso-ascii-font-family:等线;mso-fareast-font-family:
  等线;mso-hansi-font-family:等线;mso-bidi-font-family:&quot;Times New Roman&quot;;
  mso-font-kerning:0pt"><o:p></o:p></span></p>
  </td>
  <td width="123" style="width:92.1pt;border-top:none;border-left:none;
  border-bottom:solid #BFBFBF 1.0pt;border-right:solid #BFBFBF 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:15.75pt">
  <p class="MsoNormal" align="center" style="text-align:center;mso-pagination:widow-orphan"><span lang="EN-US" style="font-size:10.0pt;font-family:&quot;Arial&quot;,sans-serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt">2 -- patch attack</span><span lang="EN-US" style="mso-bidi-font-size:10.5pt;mso-ascii-font-family:等线;mso-fareast-font-family:
  等线;mso-hansi-font-family:等线;mso-bidi-font-family:&quot;Times New Roman&quot;;
  mso-font-kerning:0pt"><o:p></o:p></span></p>
  </td>
  <td width="76" style="width:2.0cm;border-top:none;border-left:none;border-bottom:
  solid #BFBFBF 1.0pt;border-right:solid #BFBFBF 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:15.75pt"></td>
  <td width="170" style="width:127.6pt;border-top:none;border-left:none;
  border-bottom:solid #BFBFBF 1.0pt;border-right:solid #BFBFBF 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:15.75pt">
  <p class="MsoNormal" align="center" style="text-align:center;mso-pagination:widow-orphan"><span lang="EN-US" style="font-size:10.0pt;font-family:&quot;Arial&quot;,sans-serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt">R50 -- Resnet50</span><span lang="EN-US" style="mso-bidi-font-size:10.5pt;mso-ascii-font-family:等线;mso-fareast-font-family:
  等线;mso-hansi-font-family:等线;mso-bidi-font-family:&quot;Times New Roman&quot;;
  mso-font-kerning:0pt"><o:p></o:p></span></p>
  </td>
  <td width="161" style="width:120.5pt;border-top:none;border-left:none;
  border-bottom:solid #BFBFBF 1.0pt;border-right:solid #BFBFBF 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:15.75pt">
  <p class="MsoNormal" align="center" style="text-align:center;mso-pagination:widow-orphan"><span lang="EN-US" style="font-size:10.0pt;font-family:&quot;Arial&quot;,sans-serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt">QB -- Query based Black Box</span><span lang="EN-US" style="mso-bidi-font-size:10.5pt;mso-ascii-font-family:等线;mso-fareast-font-family:
  等线;mso-hansi-font-family:等线;mso-bidi-font-family:&quot;Times New Roman&quot;;
  mso-font-kerning:0pt"><o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:6;height:15.75pt">
  <td width="141" style="width:105.95pt;border:solid #BFBFBF 1.0pt;border-top:
  none;padding:0cm 5.4pt 0cm 5.4pt;height:15.75pt"></td>
  <td width="123" style="width:92.1pt;border-top:none;border-left:none;
  border-bottom:solid #BFBFBF 1.0pt;border-right:solid #BFBFBF 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:15.75pt">
  <p class="MsoNormal" align="center" style="text-align:center;mso-pagination:widow-orphan"><span lang="EN-US" style="font-size:10.0pt;font-family:&quot;Arial&quot;,sans-serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt">3 -- trojan attack</span><span lang="EN-US" style="mso-bidi-font-size:10.5pt;mso-ascii-font-family:等线;mso-fareast-font-family:
  等线;mso-hansi-font-family:等线;mso-bidi-font-family:&quot;Times New Roman&quot;;
  mso-font-kerning:0pt"><o:p></o:p></span></p>
  </td>
  <td width="76" style="width:2.0cm;border-top:none;border-left:none;border-bottom:
  solid #BFBFBF 1.0pt;border-right:solid #BFBFBF 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:15.75pt"></td>
  <td width="170" style="width:127.6pt;border-top:none;border-left:none;
  border-bottom:solid #BFBFBF 1.0pt;border-right:solid #BFBFBF 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:15.75pt">
  <p class="MsoNormal" align="center" style="text-align:center;mso-pagination:widow-orphan"><span lang="EN-US" style="font-size:10.0pt;font-family:&quot;Arial&quot;,sans-serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt">R18 -- Resnet18</span><span lang="EN-US" style="mso-bidi-font-size:10.5pt;mso-ascii-font-family:等线;mso-fareast-font-family:
  等线;mso-hansi-font-family:等线;mso-bidi-font-family:&quot;Times New Roman&quot;;
  mso-font-kerning:0pt"><o:p></o:p></span></p>
  </td>
  <td width="161" style="width:120.5pt;border-top:none;border-left:none;
  border-bottom:solid #BFBFBF 1.0pt;border-right:solid #BFBFBF 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:15.75pt"></td>
 </tr>
 <tr style="mso-yfti-irow:7;height:15.75pt">
  <td width="141" style="width:105.95pt;border:solid #BFBFBF 1.0pt;border-top:
  none;padding:0cm 5.4pt 0cm 5.4pt;height:15.75pt"></td>
  <td width="123" style="width:92.1pt;border-top:none;border-left:none;
  border-bottom:solid #BFBFBF 1.0pt;border-right:solid #BFBFBF 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:15.75pt"></td>
  <td width="76" style="width:2.0cm;border-top:none;border-left:none;border-bottom:
  solid #BFBFBF 1.0pt;border-right:solid #BFBFBF 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:15.75pt"></td>
  <td width="170" style="width:127.6pt;border-top:none;border-left:none;
  border-bottom:solid #BFBFBF 1.0pt;border-right:solid #BFBFBF 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:15.75pt">
  <p class="MsoNormal" align="center" style="text-align:center;mso-pagination:widow-orphan"><span lang="EN-US" style="font-size:10.0pt;font-family:&quot;Arial&quot;,sans-serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt">L04 –&nbsp;<span class="SpellE">Lenet</span>&nbsp;with
  4&nbsp;<span class="SpellE">convs</span></span><span lang="EN-US" style="mso-bidi-font-size:10.5pt;mso-ascii-font-family:等线;mso-fareast-font-family:
  等线;mso-hansi-font-family:等线;mso-bidi-font-family:&quot;Times New Roman&quot;;
  mso-font-kerning:0pt"><o:p></o:p></span></p>
  </td>
  <td width="161" style="width:120.5pt;border-top:none;border-left:none;
  border-bottom:solid #BFBFBF 1.0pt;border-right:solid #BFBFBF 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:15.75pt"></td>
 </tr>
 <tr style="mso-yfti-irow:8;mso-yfti-lastrow:yes;height:15.75pt">
  <td width="141" style="width:105.95pt;border:solid #BFBFBF 1.0pt;border-top:
  none;padding:0cm 5.4pt 0cm 5.4pt;height:15.75pt"></td>
  <td width="123" style="width:92.1pt;border-top:none;border-left:none;
  border-bottom:solid #BFBFBF 1.0pt;border-right:solid #BFBFBF 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:15.75pt"></td>
  <td width="76" style="width:2.0cm;border-top:none;border-left:none;border-bottom:
  solid #BFBFBF 1.0pt;border-right:solid #BFBFBF 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:15.75pt"></td>
  <td width="170" style="width:127.6pt;border-top:none;border-left:none;
  border-bottom:solid #BFBFBF 1.0pt;border-right:solid #BFBFBF 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:15.75pt">
  <p class="MsoNormal" align="center" style="text-align:center;mso-pagination:widow-orphan"><span lang="EN-US" style="font-size:10.0pt;font-family:&quot;Arial&quot;,sans-serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt">I03 -- inception3</span><span lang="EN-US" style="mso-bidi-font-size:10.5pt;mso-ascii-font-family:等线;mso-fareast-font-family:
  等线;mso-hansi-font-family:等线;mso-bidi-font-family:&quot;Times New Roman&quot;;
  mso-font-kerning:0pt"><o:p></o:p></span></p>
  </td>
  <td width="161" style="width:120.5pt;border-top:none;border-left:none;
  border-bottom:solid #BFBFBF 1.0pt;border-right:solid #BFBFBF 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:15.75pt"></td>
 </tr>
</tbody></table>

<p class="MsoNormal" style="mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;mso-ascii-font-family:等线;mso-fareast-font-family:
等线;mso-hansi-font-family:等线;mso-bidi-font-family:宋体;color:black;mso-font-kerning:
0pt">&nbsp;<o:p></o:p></span></p>

<p class="MsoNormal" style="mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;mso-ascii-font-family:等线;mso-fareast-font-family:
等线;mso-hansi-font-family:等线;mso-bidi-font-family:宋体;color:black;mso-font-kerning:
0pt">&nbsp;<o:p></o:p></span></p>

<p class="MsoNormal"><span lang="EN-US"><o:p>&nbsp;</o:p></span></p>

</div>




</body>
         
     
         
