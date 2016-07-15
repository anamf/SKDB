/* Open source system for classification learning from very large data
** Copyright (C) 2012 Geoffrey I Webb
**
** This program is free software: you can redistribute it and/or modify
** it under the terms of the GNU General Public License as published by
** the Free Software Foundation, either version 3 of the License, or
** (at your option) any later version.
** 
** This program is distributed in the hope that it will be useful,
** but WITHOUT ANY WARRANTY; without even the implied warranty of
** MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
** GNU General Public License for more details.
** 
** You should have received a copy of the GNU General Public License
** along with this program. If not, see <http://www.gnu.org/licenses/>.
**
** Please report any bugs to Geoff Webb <geoff.webb@monash.edu>
*/
#pragma once

#include <string>
#include <vector>

#include "instanceStream.h"
#include "capabilities.h"

/**
 <!-- globalinfo-start -->
 * Generic class for a learner/classifier.<br/>
 <!-- globalinfo-end -->
 * 
 * @author Geoff Webb (geoff.webb@monash.edu)
 */

class learner
{
public:
  learner();
  virtual ~learner(void);

/** Moved to IncrementalLearner
  virtual void reset(InstanceStream &is) = 0;   ///< reset the learner prior to training
  virtual void initialisePass() = 0;            ///< must be called to initialise a pass through an instance stream before calling train(const instance). should not be used with train(InstanceStream)
  virtual void train(const instance &inst) = 0; ///< primary training method. train from a single instance. used in conjunction with initialisePass and finalisePass
  virtual void finalisePass() = 0;              ///< must be called to finalise a pass through an instance stream using train(const instance). should not be used with train(InstanceStream)
  virtual bool trainingIsFinished() = 0;        ///< true iff no more passes are required. updated by finalisePass()
**/

  virtual void train(InstanceStream &is) = 0;  ///< train the classifier from an instance stream

  virtual void classify(const instance &inst, std::vector<double> &classDist) = 0;  ///< infer the class distribution for the current instance in the instance stream

  virtual void getCapabilities(capabilities &c) = 0; ///< describes what kind of data the learner is able to handle
  
  void testCapabilities(InstanceStream &is);         ///< test whether the learner is able to handle the data.
  
  virtual void printClassifier() {};            ///< print details of the classifier that has been created

  inline std::string* getName() { return &name_; } ///< return the learner's name

protected:
  std::string name_;  ///< the learner's name
};
