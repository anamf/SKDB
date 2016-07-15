#pragma once
#include "instanceStreamFilter.h"
#include "utils.h"

class FilterSet : public ptrVec<InstanceStreamFilter> {
public:
  FilterSet(void);
  ~FilterSet();

  InstanceStream* apply(InstanceStream *source); ///< create a new InstanceStream by applying the set of filters
};
