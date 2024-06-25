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
    {"text": "The sun dips below the horizon, painting the sky orange."}, {"text": "A gentle breeze whispers through the autumn leaves."}, 
    {"text": "The moon casts a silver glow on the tranquil lake."}, {"text": "A solitary lighthouse stands guard on the rocky shore."}, 
    {"text": "The city awakens as morning light filters through the streets."}, {"text": "Stars twinkle in the velvety blanket of the night sky."}, 
    {"text": "The aroma of fresh coffee fills the cozy kitchen."}, {"text": "A curious kitten pounces on a fluttering butterfly."}, 
    {"text": "The sun dips below the horizon, painting the sky orange."}, {"text": "A gentle breeze whispers through the autumn leaves."}, 
    {"text": "The moon casts a silver glow on the tranquil lake."}, {"text": "A solitary lighthouse stands guard on the rocky shore."}, 
    {"text": "The city awakens as morning light filters through the streets."}, {"text": "Stars twinkle in the velvety blanket of the night sky."}, 
    {"text": "The aroma of fresh coffee fills the cozy kitchen."}, {"text": "A curious kitten pounces on a fluttering butterfly.The sun dips below the horizon, painting the sky orange."}, 
    {"text": "A gentle breeze whispers through the autumn leaves."}, {"text": "The moon casts a silver glow on the tranquil lake."}, {"text": "A solitary lighthouse stands guard on the rocky shore."}
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
    "The sun dips below the horizon, painting the sky orange.",
    "A gentle breeze whispers through the autumn leaves.",
    "The moon casts a silver glow on the tranquil lake.",
    "A solitary lighthouse stands guard on the rocky shore.",
    "The city awakens as morning light filters through the streets.",
    "Stars twinkle in the velvety blanket of the night sky.",
    "The aroma of fresh coffee fills the cozy kitchen.",
    "A curious kitten pounces on a fluttering butterfly.",
    "The sun dips below the horizon, painting the sky orange.",
    "A gentle breeze whispers through the autumn leaves.",
    "The moon casts a silver glow on the tranquil lake.",
    "A solitary lighthouse stands guard on the rocky shore.",
    "The city awakens as morning light filters through the streets.",
    "Stars twinkle in the velvety blanket of the night sky.",
    "The aroma of fresh coffee fills the cozy kitchen.",
    "A curious kitten pounces on a fluttering butterfly."
    "The sun dips below the horizon, painting the sky orange.",
    "A gentle breeze whispers through the autumn leaves.",
    "The moon casts a silver glow on the tranquil lake.",
    "A solitary lighthouse stands guard on the rocky shore.",
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