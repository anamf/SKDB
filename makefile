CC      = g++
CFLAGS  = -O3 -DNDEBUG
SOURCE  = gigal.cpp kdbSelective.cpp kdb.cpp aode.cpp tan.cpp nb.cpp incrementalLearner.cpp learner.cpp correlationMeasures.cpp globals.cpp utils.cpp instanceStream.cpp instance.cpp capabilities.cpp distributionTree.cpp mtrand.cpp ALGLIB_specialfunctions.cpp xxyDist.cpp xyDist.cpp yDist.cpp ALGLIB_ap.cpp alglibinternal.cpp learnerRegistry.cpp instanceFile.cpp instanceStreamDiscretiser.cpp discretiser.cpp instanceStreamClassFilter.cpp FilterSet.cpp trainTest.cpp xVal.cpp eqDepthDiscretiser.cpp MDLDiscretiser.cpp xValInstanceStream.cpp instanceStreamFilter.cpp
default: gigal  

depend: .depend

.depend: $(SOURCE)
	rm -f ./.depend
	$(CC) $(CFLAGS) -MM $^ >> ./.depend;

include .depend

gigal: ${SOURCE}
	$(CC) -o $@ ${SOURCE} $(CFLAGS)

gigal64: ${SOURCE}
	$(CC) -o $@ ${SOURCE} $(CFLAGS) -DSIXTYFOURBITCOUNTS
