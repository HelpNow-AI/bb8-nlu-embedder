import os
import json
import logging
import logging.config

# Logger
with open('./_config/logger.json', 'r') as f:
        config = json.load(f)
logging.config.dictConfig(config)
logger = logging.getLogger("bb8-nlu-loggger")


if "NAMESPACE" in os.environ:
    if os.environ['NAMESPACE'] == 'bb8-sandbox':
        nlu_server = 'bb8-nlu-sandbox.dev.opsnow.com'
        logger.setLevel('DEBUG')
    elif os.environ['NAMESPACE'] == 'bb8-dev':
        nlu_server = 'bb8-nlu-dev.dev.opsnow.com'
        logger.setLevel('DEBUG')
    elif os.environ['NAMESPACE'] == 'bb8-prod':
        nlu_server = 'bb8-nlu-prod.okc1.opsnow.com'
        logger.setLevel('INFO')
else:
    logger.setLevel('INFO')
    nlu_server = 'localhost:8000'