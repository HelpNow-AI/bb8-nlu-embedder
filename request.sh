curl -X 'POST' \
  'http://localhost:3000/embedding_batch' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "sentences": [
    "hello, my name is jaehyeong.", "I really want to go home.", "Sorry.. But, I hate you.", "hahahahahah, blablablabla", 
    "hello, my name is jaehyeong.", "I really want to go home.", "Sorry.. But, I hate you.", "hahahahahah, blablablabla",
    "hello, my name is jaehyeong.", "I really want to go home.", "Sorry.. But, I hate you.", "hahahahahah, blablablabla",
    "hello, my name is jaehyeong.", "I really want to go home.", "Sorry.. But, I hate you.", "hahahahahah, blablablabla",
    "hello, my name is jaehyeong.", "I really want to go home.", "Sorry.. But, I hate you.", "hahahahahah, blablablabla"
  ]

}'



curl -X 'POST' \
  'http://localhost:3000/rerank' \
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
}'