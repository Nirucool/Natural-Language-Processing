This folder has 3 corpus folders
1. corpus folder is for given data
2. corpus-process folder is corpus for individual files processed for cfg or dependency parse of individual files.
3. corpus-out folder contains individual files after dependency parse for each file.

output1 folder--->
It contains->All parsing is after preprocessing and contains parsing/tagging of lines<=50
1. cfgParse.txt contains CFG parse for entire corpus.
2. dependencyParse.txt contains dependency parse for entire corpus.
3. posTagged.txt contains pos tagging of entire corpus.

version of Stanford Parser-
stanford-parser-full-2017-06-09     
It is 3.8.0 version

4. data.txt is preprocessed data from all 100 files.
5. prepCountPerFile.txt contains preposition count in every file.
6. commonPreposition.txt contains 3 most common prepositions in the entire corpus.


Question 1.4.pdf for Question 1.4 answer.
Question1.py is Python file for 1.1-1.3

steps to run the code:
 Copy all folder explained above along with stanford-parser-full-2017-06-09 folder and Question1.py inside Scripts folder of PycharmProject.

For creating the parsing and tagging files-Windows-command line....
move inside stanford-parser-full-2017-06-09 folder and add all files to classpath using
java -cp "*" or java -cp "*;" depends on windows version.
Then run below commands---

# For dependency parse
java -Xmx12g -cp "*" edu.stanford.nlp.parser.lexparser.LexicalizedParser -outputFormat "typedDependencies" edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz data.txt >>dependencyParse.txt
# For CFG parse
java -Xmx12g -cp "*" edu.stanford.nlp.parser.lexparser.LexicalizedParser -outputFormat "penn" edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz data.txt >>cfgParse.txt
# For pos tagging
java -Xmx12g -cp "*" edu.stanford.nlp.parser.lexparser.LexicalizedParser -outputFormat "wordsAndTags" edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz data.txt >>posTagged.txt

For processing files individually for constituent and dependency parse use commands for penn and typeDependencies given above in a Windows batch file or individually for each file.

For running 1.1-1.3 simply run Question1.py

