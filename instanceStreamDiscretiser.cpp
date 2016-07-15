#include "instanceStreamDiscretiser.h"
#include "globals.h"
#include "mtrand.h"
#include "utils.h"

// discretisers
#include "eqDepthDiscretiser.h"
#include "MDLDiscretiser.h"

InstanceStreamDiscretiser::InstanceStreamDiscretiser(const char* name,
		char* const *& argv, char* const * end) {
	targetSampleSize_ = 100000;

	if (streq(name, "equal-depth") || streq(name, "equal-frequency")) {
		theDiscretiser = new eqDepthDiscretiser(argv, end);
	} else if (streq(name, "mdl")) {
		theDiscretiser = new MDLDiscretiser(argv, end);
	}else {
		error("Discretiser %s not supported", name);
	}
	printMetaData_ = false;
	allNumWithMiss_ = false;
	// get arguments
	while (argv != end) {
		if (**argv == '-' && argv[0][1] == 's') {
			getUIntFromStr(argv[0] + 2, targetSampleSize_, "s");
			++argv;
		} else if (**argv == '-' && argv[0][1] == 'M') {
			allNumWithMiss_ = true;
			++argv;
		} else if (**argv == '-' && argv[0][1] == 'P') {
			printMetaData_ = true;
			++argv;
		} else if (argv[0][1] == 'o') {
			filename_ = &argv[0][2];
			++argv;
		} else {
			break;  // do not consume the remaining arguments
		}
	}

	if (printMetaData_ == true && filename_ == NULL) {
		error("The meta file for output must be specified by -o.\n");
	}

	if (theDiscretiser == NULL) {
		theDiscretiser = new eqDepthDiscretiser(argv, end);
	}

}

InstanceStreamDiscretiser::~InstanceStreamDiscretiser(void) {
	delete theDiscretiser;
}

/// set the source for the filter
void InstanceStreamDiscretiser::setSource(InstanceStream &src) {
	std::vector<std::vector<NumValue> > vals;
	std::vector<CatValue> classes;
	InstanceCount count;
	MTRand_int32 rand;

	source_ = &src;
	metaData_.setSource(src.getMetaData());

	if (allNumWithMiss_)
		metaData_.setAllAttsMissing();
	InstanceStream::metaData_ = &metaData_;

	// get the sample
	vals.resize(src.getNoNumAtts());
	for (NumericAttribute a = 0; a < src.getNoNumAtts(); a++) {
		vals[a].resize(targetSampleSize_);
	}
	classes.resize(targetSampleSize_);

	sourceInst_.init(src);
	count = 0;

	src.rewind();

	instance inst(src);

	while (src.advance(inst)) {
		count++;  // keep track of the number of values seen
		unsigned long index;

		if (count <= targetSampleSize_) {
			// if we have not yet got targetSampleSize_ examples, add this value to the end
			index = count - 1;
		} else {
			// otherwise randomly determine whether to insert the value and where
			index = rand(count);
		}

		if (index < targetSampleSize_) {
			classes[index] = inst.getClass();
			for (NumericAttribute a = 0; a < src.getNoNumAtts(); a++) {
				vals[a][index] = inst.getNumVal(a);
			}
		}
	}

	metaData_.cuts.resize(src.getNoNumAtts());
	metaData_.discAttValNames_.resize(src.getNoNumAtts());

	if (classes.size() > count)
		classes.resize(count);

	char buf[200];

	FILE * output;
	if (printMetaData_) {
		output = fopen(filename_, "w");
	}

	if (printMetaData_) {

		for (CategoricalAttribute ca = 0; ca < src.getNoCatAtts(); ca++) {
			fprintf(output, "%s: ", src.getCatAttName(ca));
			if (verbosity >= 2)
				printf("%s: ", src.getCatAttName(ca));

			for (unsigned int i = 0; i < src.getNoValues(ca); i++) {

				if (i !=  (src.getNoValues(ca)-1))
					sprintf(buf, "%u, ", i);
				else
					sprintf(buf, "%u", i);

				fprintf(output, "%s", buf);
				if (verbosity >= 2) {
					printf("%s", buf);
				}
			}

		        fprintf(output, "\n");
			if (verbosity >= 2)
			  putchar('\n');

		}



		for (NumericAttribute na = 0; na < src.getNoNumAtts(); na++) {
			if (vals[na].size() > count) {
				// if fewer values than stored, truncate
				vals[na].resize(count);
			}
			metaData_.cuts[na].clear();
			// discretise then set up the value names
			theDiscretiser->discretise(vals[na], classes, src.getNoClasses(),
					metaData_.cuts[na]);

			fprintf(output, "%s: ", src.getNumAttName(na));
			if (verbosity >= 2)
				printf("%s: ", src.getNumAttName(na));

                        unsigned int i;
			// loop through all of the intervals that have been formed (note, one more interval than cut)
			for (i = 0; i <= metaData_.cuts[na].size(); i++) {

				//sprintf(buf, "%u", i);
				if (i != metaData_.cuts[na].size())
					sprintf(buf, "%u, ", i);
				else
					sprintf(buf, "%u", i);

				fprintf(output, "%s", buf);
				if (verbosity >= 2) {
					printf("%s", buf);
				}
			}

			// add missing value
			if (metaData_.hasNumMissing(na)) {
                                sprintf(buf, ", %u ", i++);
				fprintf(output, "%s\n", buf);
				if (verbosity >= 2)
					printf("%s\n", buf);

			} else {
				fprintf(output, "\n");
				if (verbosity >= 2)
					putchar('\n');
			}

		}



		fprintf(output, "class: ");
		const char *sep = "";
		for (CatValue i = 0; i < src.getNoClasses(); i++) {
			fprintf(output, "%s%u", sep, i);
			sep = ", ";
		}
		fprintf(output, "\n");

	}

	else {

		for (NumericAttribute na = 0; na < src.getNoNumAtts(); na++) {
			if (vals[na].size() > count) {
				// if fewer values than stored, truncate
				vals[na].resize(count);
			}
			metaData_.cuts[na].clear();
			// discretise then set up the value names
			theDiscretiser->discretise(vals[na], classes, src.getNoClasses(),
					metaData_.cuts[na]);
			if (verbosity >= 3)
				printf("Discretisation for %s: ", src.getNumAttName(na));
			// loop through all of the intervals that have been formed (note, one more interval than cut)
			for (unsigned int i = 0; i <= metaData_.cuts[na].size(); i++) {
				if (i == 0) {
					if (metaData_.cuts[na].size() == 0) {
						sprintf(buf, "known");
					} else {
						// print values to precision+1 in case cut points are placed between actual values
						sprintf(buf, "<=%.*f", src.getPrecision(na) + 1,
								metaData_.cuts[na][i]);
					}
				} else if (i < metaData_.cuts[na].size()) {
					// print values to precision+1 in case cut points are placed between actual values
					sprintf(buf, "(%.*f,%.*f]", src.getPrecision(na) + 1,
							metaData_.cuts[na][i - 1], src.getPrecision(na) + 1,
							metaData_.cuts[na][i]);
				} else {
					// print values to precision+1 in case cut points are placed between actual values
					sprintf(buf, ">%.*f", src.getPrecision(na) + 1,
							metaData_.cuts[na][i - 1]);
				}
				if (verbosity >= 3) {
					printf("%s ", buf);
					if (i != metaData_.cuts[na].size())
						printf(", ");
				}
				char *newbuf;
				safeAlloc(newbuf, strlen(buf) + 1);
				strcpy(newbuf, buf);
				metaData_.discAttValNames_[na].push_back(newbuf);

			}
			// add missing value
			if (metaData_.hasNumMissing(na)) {
				const char *mstr;
				mstr = ", unknown";
				char *newbuf;
				if (verbosity >= 3)
					printf("%s\n", mstr);
				safeAlloc(newbuf, strlen(mstr) + 1);
				strcpy(newbuf, mstr);
				metaData_.discAttValNames_[na].push_back(newbuf);
			} else {
				if (verbosity >= 3)
					putchar('\n');
			}
		}

	}

	if (printMetaData_) {
		fclose(output);
	}
	rewind();
}

/// return to the first instance in the stream
void InstanceStreamDiscretiser::rewind() {
	source_->rewind();
}

/// advance, discarding the next instance in the stream.  Return true iff successful.
bool InstanceStreamDiscretiser::advance() {
	return source_->advance();
}

CatValue InstanceStreamDiscretiser::discretise(const NumValue val,

		const NumericAttribute na) const {

	if (val == MISSINGNUM) {
		return metaData_.cuts[na].size() + 1;
	} else if (metaData_.cuts[na].size() == 0) {
		return 0;
	} else if (val > metaData_.cuts[na].back()) {
		return metaData_.cuts[na].size();
	} else {
		unsigned int upper = metaData_.cuts[na].size() - 1;
		unsigned int lower = 0;

		while (upper > lower) {
			const unsigned int mid = lower + (upper - lower) / 2;

			if (val <= metaData_.cuts[na][mid]) {
				upper = mid;
			} else {
				lower = mid + 1;
			}
		}

		assert(upper == lower);
		return upper;
	}
}

void InstanceStreamDiscretiser::discretiseInstance(const instance &inst,
		instance &instDisc) {
	CategoricalAttribute ca;

	for (ca = 0; ca < source_->getNoCatAtts(); ca++) {
		setCatVal(instDisc, ca, inst.getCatVal(ca));
	}

	for (NumericAttribute na = 0; na < source_->getNoNumAtts(); na++) {
		setCatVal(instDisc, ca, discretise(inst.getNumVal(na), na));
		ca++;
	}
	instDisc.setClass(inst.getClass());
}

/// advance to the next instance in the stream. Return true iff successful. @param inst the instance record to receive the new instance. 
bool InstanceStreamDiscretiser::advance(instance &inst) {

	if (!source_->advance(sourceInst_))
		return false;

	setClass(inst, sourceInst_.getClass());

	CategoricalAttribute ca;

	for (ca = 0; ca < source_->getNoCatAtts(); ca++) {
		setCatVal(inst, ca, sourceInst_.getCatVal(ca));
	}

	NumericAttribute na;
	for (na = 0; na < source_->getNoNumAtts(); na++) {
		setCatVal(inst, ca, discretise(sourceInst_.getNumVal(na), na));
		ca++;
	}

	return true;
}

bool InstanceStreamDiscretiser::advanceNumeric(instance &inst) {

	inst.init(*source_);
	if (!source_->advance(sourceInst_))
		return false;

	setClass(inst, sourceInst_.getClass());

	CategoricalAttribute ca;

	for (ca = 0; ca < source_->getNoCatAtts(); ca++) {
		setCatVal(inst, ca, sourceInst_.getCatVal(ca));
	}

	for (NumericAttribute na = 0; na < source_->getNoNumAtts(); na++) {
		setNumVal(inst, na, sourceInst_.getNumVal(na));
	}

	return true;
}

/// true if we have advanced past the last instance
bool InstanceStreamDiscretiser::isAtEnd() {
	return source_->isAtEnd();
}

/// the number of instances in the stream. This may require a pass through the stream to determine so should be used only if absolutely necessary.
InstanceCount InstanceStreamDiscretiser::size() {
	return source_->size();
}

void InstanceStreamDiscretiser::MetaData::setSource(
		InstanceStream::MetaData* source) {
	source_ = source;
	noOfSourceCatAtts_ = source->getNoCatAtts();
}

/// return the number of categorical attributes
unsigned int InstanceStreamDiscretiser::MetaData::getNoCatAtts() const {
	return source_->getNoCatAtts() + cuts.size();
}

/// return the number of values for a categorical attribute
unsigned int InstanceStreamDiscretiser::MetaData::getNoValues(
		CategoricalAttribute att) const {
	if (att < source_->getNoCatAtts())
		return source_->getNoValues(att);
	else {
		return cuts[att - source_->getNoCatAtts()].size() + 1
				+ source_->hasNumMissing(att - source_->getNoCatAtts());
	}
}

/// return the name for a categorical Attribute
const char* InstanceStreamDiscretiser::MetaData::getCatAttName(
		CategoricalAttribute att) const {
	if (att < source_->getNoCatAtts())
		return source_->getCatAttName(att);
	else
		return source_->getNumAttName(att - source_->getNoCatAtts());
}

/// return the name for a categorical attribute value
const char* InstanceStreamDiscretiser::MetaData::getCatAttValName(
		CategoricalAttribute att, CatValue val) const {
	if (att < source_->getNoCatAtts())
		return source_->getCatAttValName(att, val);
	else
		return discAttValNames_[att - source_->getNoCatAtts()][val];
}

/// return the number of numeric attributes
unsigned int InstanceStreamDiscretiser::MetaData::getNoNumAtts() const {
	return 0;
}

unsigned int InstanceStreamDiscretiser::MetaData::getNoOrigCatAtts() const {
	return noOfSourceCatAtts_;
}

/// return the name for a numeric attribute
const char* InstanceStreamDiscretiser::MetaData::getNumAttName(
		CategoricalAttribute) const {
	assert(false);
	return NULL;
}

/// return the name for a numeric attribute
unsigned int InstanceStreamDiscretiser::MetaData::getPrecision(
		NumericAttribute) const {
	assert(false);
	return 0;
}

/// return a string that gives a meaningful name for the stream
const char* InstanceStreamDiscretiser::MetaData::getName() {
	return "Discretised instance stream";
}

int InstanceStreamDiscretiser::MetaData::hasCatMissing(
		CategoricalAttribute att) const {
	return source_->hasCatMissing(att);
}

int InstanceStreamDiscretiser::MetaData::hasNumMissing(
		NumericAttribute att) const {
	return source_->hasNumMissing(att);
}

void InstanceStreamDiscretiser::MetaData::setAllAttsMissing() {
	source_->setAllAttsMissing();
}
