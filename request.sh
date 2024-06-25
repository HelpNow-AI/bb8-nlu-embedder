#!/bin/sh

curl -X 'POST' \
  'http://localhost:3000/api/nlu/sentence-embedding' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "query": "hello, my name is jaehyeong."
}' > /dev/null 2>&1

if [ $? -eq 0 ]; then
  echo "'/api/nlu/sentence-embedding' > 🟢"
else
  echo "'/api/nlu/sentence-embedding' > 🔴"
fi


curl -X 'POST' \
  'http://localhost:3000/api/nlu/sentence-embedding-batch' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "data": [
    {"text": "hello, my name is jaehyeong."}, {"text": "I really want to go home."}, {"text": "Sorry.. But, I hate you."}, {"text": "hahahahahah, blablablabla"}, 
    {"text": "hello, my name is jaehyeong."}, {"text": "I really want to go home."}, {"text": "Sorry.. But, I hate you."}, {"text": "hahahahahah, blablablabla"},
    {"text": "hello, my name is jaehyeong."}, {"text": "I really want to go home."}, {"text": "Sorry.. But, I hate you."}, {"text": "hahahahahah, blablablabla"},
    {"text": "hello, my name is jaehyeong."}, {"text": "I really want to go home."}, {"text": "Sorry.. But, I hate you."}, {"text": "hahahahahah, blablablabla"},
    {"text": "hello, my name is jaehyeong."}, {"text": "I really want to go home."}, {"text": "Sorry.. But, I hate you."}, {"text": "hahahahahah, blablablabla"}
  ]
}' > /dev/null 2>&1 

if [ $? -eq 0 ]; then
  echo "'/api/nlu/sentence-embedding-batch' > 🟢"
else
  echo "'/api/nlu/sentence-embedding-batch' > 🔴"
fi



curl -X 'POST' \
  'http://localhost:3000/api/assist/sentence-embedding' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "query": "hello, my name is jaehyeong."
}' > /dev/null 2>&1

if [ $? -eq 0 ]; then
  echo "'/api/assist/sentence-embedding' > 🟢"
else
  echo "'/api/assist/sentence-embedding' > 🔴"
fi


curl -X 'POST' \
  'http://localhost:3000/api/assist/sentence-embedding-batch' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "data": [
    "hello, my name is jaehyeong.", "I really want to go home.", "Sorry.. But, I hate you.", "hahahahahah, blablablabla", 
    "hello, my name is jaehyeong.", "I really want to go home.", "Sorry.. But, I hate you.", "hahahahahah, blablablabla",
    "hello, my name is jaehyeong.", "I really want to go home.", "Sorry.. But, I hate you.", "hahahahahah, blablablabla",
    "hello, my name is jaehyeong.", "I really want to go home.", "Sorry.. But, I hate you.", "hahahahahah, blablablabla",
    "hello, my name is jaehyeong.", "I really want to go home.", "Sorry.. But, I hate you.", "hahahahahah, blablablabla"
  ]
}' > /dev/null 2>&1

if [ $? -eq 0 ]; then
  echo "'/api/assist/sentence-embedding-batch' > 🟢"
else
  echo "'/api/assist/sentence-embedding-batch' > 🔴"
fi


curl -X 'POST' \
  'http://localhost:3000/api/assist/cross-encoder/similarity-scores' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "data": [
    {"query": "This is a apple", "passage": "This is a banana"},
    {"query": "I really want to go home.", "passage": "Sorry.. But, I hate you."},
    {"query": "How can I go to police station?", "passage": "Oh, Police station is over there."},
    {"query": "This is a query", "passage": "This is a passage"},
    {"query": "This is a query", "passage": "This is a passage"},
    {"query": "This is a query", "passage": "This is a passage"},
    {"query": "This is a query", "passage": "This is a passage"},
    {"query": "This is a query", "passage": "This is a passage"},
    {"query": "This is a query", "passage": "This is a passage"},
    {"query": "This is a query", "passage": "This is a query"}
  ]
}' > /dev/null 2>&1

if [ $? -eq 0 ]; then
  echo "'/api/assist/cross-encoder/similarity-scores' > 🟢"
else
  echo "'/api/assist/cross-encoder/similarity-scores' > 🔴"
fi