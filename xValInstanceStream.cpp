#include "xValInstanceStream.h"

XValInstanceStream::XValInstanceStream(InstanceStream *source, const unsigned int noOfFolds, const unsigned int seed)
  : source_(source), noOfFolds_(noOfFolds), seed_(seed)
{ metaData_ = source->getMetaData();
  startSubstream(0, true);
}

XValInstanceStream::~XValInstanceStream(void)
{
}

/// start training or testing for a new fold
void XValInstanceStream::startSubstream(const unsigned int fold, const bool training) {
  fold_ = fold;
  training_ = training;
  rewind();
}

/// return to the first instance in the stream
void XValInstanceStream::rewind() {
  source_->rewind();
  rand_.seed(seed_);
  count_ = 0;
}

/// advance, discarding the next instance in the stream.  Return true iff successful.
bool XValInstanceStream::advance() {
  while (source_->advance()) {
    if (rand_() % noOfFolds_ == fold_) {
      // test instance
      if (!training_) {
        count_++;
        return true;
      }
    }
    else {
      // training instance
      if (training_) {
        count_++;
        return true;
      }
    }
  }
  return false;
}

/// advance to the next instance in the stream.Return true iff successful. @param inst the instance record to receive the new instance. 
bool XValInstanceStream::advance(instance &inst) {
  while (!source_->isAtEnd()) {
    if (rand_() % noOfFolds_ == fold_) {
      // test instance
      if (training_) {
        source_->advance();
      }
      else {
        // testing
        if (source_->advance(inst)) {
          count_++;
          return true;
        }
        else return false;
      }
    }
    else {
      // training instance
      if (training_) {
        if (source_->advance(inst)) {
          count_++;
          return true;
        }
        else return false;
      }
      else {
        // testing
        source_->advance();
      }
    }
  }
  return false;
}

/// true if we have advanced past the last instance
bool XValInstanceStream::isAtEnd() {
  return source_->isAtEnd();
}

/// the number of instances in the stream. This may require a pass through the stream to determine so should be used only if absolutely necessary.  The stream state is undefined after a call to size(), so a rewind shouldbe performed before the next advance.
InstanceCount XValInstanceStream::size() {
  if (!isAtEnd()) {
    instance inst(*this);

    while (!isAtEnd()) advance(inst);
  }

  return count_;
}

