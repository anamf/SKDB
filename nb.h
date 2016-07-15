/* Open source system for classification learning from very large data
 * Copyright (C) 2012 Geoffrey I Webb
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 *
 * Please report any bugs to Geoff Webb <geoff.webb@monash.edu>
 */

#pragma once

#include "incrementalLearner.h"
#include "xyDist.h"
#include "xxyDist.h"
/**
<!-- globalinfo-start -->
 * Class for a Naive Bayes classifier.<br/>
 * <br/>
 * For more information on Naive Bayes classifiers, see:<br/>
 * <br/>
 * George H. John, Pat Langley: Estimating Continuous Distributions in Bayesian 
 * Classifiers. In: Eleventh Conference on Uncertainty in Artificial 
 * Intelligence, San Mateo, 338-345, 1995.
 <!-- globalinfo-end -->
 *
 <!-- technical-bibtex-start -->
 * BibTeX:
 * <pre>
 * \@inproceedings{John1995,
 *    address = {San Mateo},
 *    author = {George H. John and Pat Langley},
 *    booktitle = {Eleventh Conference on Uncertainty in Artificial 
 *                 Intelligence},
 *    pages = {338-345},
 *    publisher = {Morgan Kaufmann},
 *    title = {Estimating Continuous Distributions in Bayesian Classifiers},
 *    year = {1995}
 * }
 * </pre>
 <!-- technical-bibtex-end -->
 *
 *
 * @author Geoff Webb (geoff.webb@monash.edu)
 */
class nb : public IncrementalLearner
{
public:
  
  /**
   * @param argv Options for the NB classifier
   * @param argc Number of options for NB
   */
  nb(char*const*& argv, char*const* end);
  
  
  ~nb(void);
  
  void reset(InstanceStream &is);   ///< reset the learner prior to training
  void initialisePass();            ///< must be called to initialise a pass through an instance stream before calling train(const instance). should not be used with train(InstanceStream)
  void train(const instance &inst); ///< primary training method. train from a single instance. used in conjunction with initialisePass and finalisePass
  void finalisePass();              ///< must be called to finalise a pass through an instance stream using train(const instance). should not be used with train(InstanceStream)
  bool trainingIsFinished();        ///< true iff no more passes are required. updated by finalisePass()
  void getCapabilities(capabilities &c); 

  /**
   * Calculates the class membership probabilities for the given test instance. 
   * 
   * @param inst The instance to be classified
   * @param classDist Predicted class probability distribution
   */
  virtual void classify(const instance &inst, std::vector<double> &classDist);
  
  
private:  

  InstanceStream* instanceStream_;

  unsigned int noCatAtts_;  ///< the number of categorical attributes.
  unsigned int noClasses_;  ///< the number of classes

  bool trainingIsFinished_; ///< true iff the learner is trained
  xyDist xyDist_;           ///< the xy distribution that NB learns from the instance stream and uses for classification

};

