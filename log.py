import tensorboardX
import logging

_logger = logging.getLogger(__name__)


class SummaryWriter(tensorboardX.SummaryWriter):
    def add_log(self, log, prefix, global_step=None):
        for (key, value) in log.items():
            self.add_scalar(f'{prefix}/{key}', value, global_step=global_step)

    def add_texts(self, texts, name, prefix, global_step=None):
        log_text = '\n'.join(
            [
                '| num | text |',
                '|:---:|:-----|',
            ]
            + [
                '|{}|{}|'.format(num_text, text)
                for (num_text, text) in enumerate(texts)
            ]
        )
        self.add_text(f'{prefix}/{name}', log_text, global_step=global_step)


def print_batch(batch, logger=None):
    logger = logger or _logger
    for (key, value) in batch.items():
        logger.info(f'{key}: shape={value.shape}')