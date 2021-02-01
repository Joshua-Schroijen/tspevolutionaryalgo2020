import logging
import r0486848

if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler("stresstest.log")
    stream_handler = logging.StreamHandler()
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    a = r0486848.r0486848()
    try:
        for i in range(1000):
            a.optimize("./tour100.csv", False)
            logger.info(f"Iteration {i} complete")

    except Exception as e:
        logger.exception("Algorithm run failed")