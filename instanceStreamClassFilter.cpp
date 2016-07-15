#include "instanceStreamClassFilter.h"
#include "utils.h"

InstanceStreamClassFilter::InstanceStreamClassFilter(InstanceStream *src, const char* posClassName, char*const*& argv, char*const* end) : metaData_(src->getMetaData(), posClassName)
{
  // get arguments
  // no arguments currently supported
  //while (argv != end) {
  //  if (**argv == '-' && argv[0][1] == 's') {
  //    getUIntFromStr(argv[0]+2, targetSampleSize_, "s");
  //    ++argv;
  //  }
  //  else {
  //    break;  // do not consume the remaining arguments
  //  }
  //}
  source_ = src;
  InstanceStream::metaData_ = &metaData_;

  for (posClass_ = 0; posClass_ < src->getNoClasses(); posClass_++) {
    if (streq(posClassName, src->getClassName(posClass_), src->areNamesCaseSensitive())) break;
  }

  if (posClass_ > src->getNoClasses()) {
    error("%s is not a class", posClassName);
  }
}

InstanceStreamClassFilter::~InstanceStreamClassFilter(void)
{
}

/// return to the first instance in the stream
void InstanceStreamClassFilter::rewind() {
  source_->rewind();
}

/// advance, discarding the next instance in the stream.  Return true iff successful.
bool InstanceStreamClassFilter::advance() {
  return source_->advance();
}

/// advance to the next instance in the stream. Return true iff successful. @param inst the instance record to receive the new instance. 
bool InstanceStreamClassFilter::advance(instance &inst) {
  if (!source_->advance(inst)) return false;

  setClass(inst, inst.getClass() == posClass_);

  return true;
}

/// true if we have advanced past the last instance
bool InstanceStreamClassFilter::isAtEnd() {
  return source_->isAtEnd();
}

/// the number of instances in the stream.This may require a pass through the stream to determine so should be used only if absolutely necessary.
InstanceCount InstanceStreamClassFilter::size() {
  return source_->size();
}

InstanceStreamClassFilter::MetaData::MetaData(InstanceStream::MetaData* meta, const char* posClassname) : posClassName_(posClassname) {
  source_ = meta;
}

/// return the number of classes
unsigned int InstanceStreamClassFilter::MetaData::getNoClasses() const {
  return 2;
}

/// return the name for a class
const char* InstanceStreamClassFilter::MetaData::getClassName(CatValue y) const {
  if (y == 0) return "Negative";
  else return posClassName_;
}


/// return a string that gives a meaningful name for the stream
const char* InstanceStreamClassFilter::MetaData::getName() {
  return "Two class instance stream";
}
