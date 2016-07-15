#pragma once
#include "instanceStream.h"
#include "mtrand.h"

class XValInstanceStream :
  public InstanceStream
{
public:
  XValInstanceStream(InstanceStream *source, const unsigned int noOfFolds, const unsigned int seed = 0);
  ~XValInstanceStream(void);

  // implementation of core InstanceStream methods
  void rewind();                                              ///< return to the first instance in the stream
  bool advance();                                             ///< advance, discarding the next instance in the stream.  Return true iff successful.
  bool advance(instance &inst);                               ///< advance to the next instance in the stream.  Return true iff successful. @param inst the instance record to receive the new instance. 
  bool isAtEnd();                                             ///< true if we have advanced past the last instance
  InstanceCount size();                                       ///< the number of instances in the stream.  This may require a pass through the stream to determine so should be used only if absolutely necessary.

  // cross validation specific methods
  void startSubstream(const unsigned int fold, const bool training);      ///< start training or testing for a new fold

private:
  InstanceStream* source_;       ///< the source stream
  MTRand_int32 rand_;            ///< random number generator for selecting folds
  const unsigned int seed_;      ///< the random number seed
  const unsigned int noOfFolds_; ///< the number of folds
  unsigned int fold_;            ///< the current fold
  bool training_;                ///< true if the current pass is a training pass. If true, the stream returns instances from the training fold. If false, the stream returns instances from all folds other than the training fold.
  InstanceCount count_;          ///< a count of the number of instances in the stream
};
