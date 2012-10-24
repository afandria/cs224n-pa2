package cs224n.assignments;

import cs224n.io.PennTreebankReader;
import cs224n.ling.Tree;
import cs224n.ling.Trees;
import cs224n.parser.EnglishPennTreebankParseEvaluator;
import cs224n.util.*;
import cs224n.util.PriorityQueue;

import java.io.BufferedReader;
import java.io.IOException;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Harness for PCFG Parser project.
 *
 * @author Dan Klein
 */
public class PCFGParserTester {

	// Parser interface ===========================================================

	/**
	 * Parsers are required to map sentences to trees.  How a parser is
	 * constructed and trained is not specified.
	 */
	public static interface Parser {
		public void train(List<Tree<String>> trainTrees);
		public Tree<String> getBestParse(List<String> sentence);
	}


	// PCFGParser =================================================================

	/**
	 * The PCFG Parser you will implement.
	 */
	public static class PCFGParser implements Parser {

		private Grammar grammar;
		private Lexicon lexicon;

		public void train(List<Tree<String>> trainTrees) {
			// TODO: before you generate your grammar, the training trees
			// need to be binarized so that rules are at most binary
			//Binarize the tree.
			List<Tree<String>> binarizedTrees = new ArrayList<Tree<String>>();
			
			for( Tree<String> trainTree: trainTrees){
				Tree<String> newTree = TreeAnnotations.annotateTree(trainTree);
				binarizedTrees.add(newTree);
				
			}
			lexicon = new Lexicon(binarizedTrees);
			grammar = new Grammar(binarizedTrees);
			System.out.println("trained!!");
		}

		public Tree<String> getBestParseOld(List<String> sentence) {
			// TODO: This implements the CKY algorithm

			CounterMap<String,String> parseScores = new CounterMap<String,String>();
						
			System.out.println(sentence.toString());
			// First deal with the lexicons 
			int index =0;
			int span =1;// All spans are 1 at the lexicon level
			for (String word : sentence) {
				for (String tag : lexicon.getAllTags()) {
					double score = lexicon.scoreTagging(word, tag);
					if(score >=0.0){ // This lexicon may generate this word
					  //We use a counter map in order to store the scores for this sentence parse.
					  parseScores.setCount(  index+" "+(index+span), tag, score);
					  	
					}
					
				}
				index = index +1;
				
			}
			
			// handle unary rules now 
			HashMap<String, Triplet<Integer,String,String>> backHash =
					new HashMap<String, Triplet<Integer,String,String>>(); // hashmap to store back propation
			
			//System.out.println("Lexicons found");
			Boolean added = true;
			
			while (added){
				added = false;	
				for( index =0; index < sentence.size(); index++){
				  // For each index+ span pair, get the counter.
				  Counter<String> count = parseScores.getCounter(index+" "+ (index+span));
				  PriorityQueue<String > countAsPQ = count.asPriorityQueue();
			      while( countAsPQ.hasNext()) {
			    	 String entry = countAsPQ.next();
			      //System.out.println("I am fine here!!");
			    	 List<UnaryRule> unaryRules =grammar.getUnaryRulesByChild(entry);
			    	 for(UnaryRule rule : unaryRules){
			    		 //These are the unary rules which might give rise to the above preterminal
			    		 double prob = rule.getScore()*parseScores.getCount(index + " " +(index+span), entry);
			    	     if(prob > parseScores.getCount(index+" "+(index+span), rule.parent)) {
			    		     parseScores.setCount(index+" "+(index+span), rule.parent,prob);
			    		     backHash.put(index + " "+ (index+span) + " "+rule.parent, new Triplet<Integer, String, String>(-1, entry , null));
			    		     added = true; 
			    		 }
			    	 }
			      }
				}
			}
			//System.out.println("Lexicon unaries dealt with");
						
			// Now work with the grammar to produce higher level probabilities 
			for(span =2 ; span <= sentence.size(); span++ ){
			  for(int begin =0 ; begin <= (sentence.size()-span); begin++){
				int end = begin + span;  
				for(int split = begin +1 ; split <= end -1;split++){
					Counter<String> countLeft = parseScores.getCounter(begin+ " " + split);
					Counter<String> countRight = parseScores.getCounter(split+ " "+ end);
					//List<BinaryRule> leftRules= new ArrayList<BinaryRule>();
					HashMap<Integer, BinaryRule> leftMap = new HashMap<Integer, BinaryRule>();
					//List<BinaryRule> rightRules=new ArrayList<BinaryRule>();
					HashMap<Integer, BinaryRule> rightMap = new HashMap<Integer, BinaryRule>();

					for (String entry: countLeft.keySet()){
						for(BinaryRule rule: grammar.getBinaryRulesByLeftChild(entry)){
							if(!leftMap.containsKey(rule.hashCode())){
								leftMap.put(rule.hashCode(), rule);
							}
						}
					}
					
					for(String entry: countRight.keySet()){
						for(BinaryRule rule: grammar.getBinaryRulesByRightChild(entry)){
							if(!rightMap.containsKey(rule.hashCode())){
								rightMap.put(rule.hashCode(), rule);
							}
						}

					}
					
					//System.out.println("About to enter the rules loops");
					for(Integer ruleHash: leftMap.keySet()){	
						if(rightMap.containsKey(ruleHash)){
							BinaryRule ruleRight = rightMap.get(ruleHash);
						  double prob = ruleRight.getScore()*parseScores.getCount(begin+" "+split, ruleRight.leftChild) *parseScores.getCount(split+" "+end, ruleRight.rightChild);
						  //System.out.println(begin+" "+ end +" "+ ruleRight.parent+ " "+ prob);
						  if(prob > parseScores.getCount(begin+" "+ end, ruleRight.parent)){
							  //System.out.println(begin+" "+ end +" "+ ruleRight.parent+ " "+ prob);
							 //System.out.println("parentrule :"+ ruleRight.getParent());
						      parseScores.setCount(begin+ " "+ end, ruleRight.getParent(),prob);
							  backHash.put(begin+" "+end+ " "+ruleRight.parent, new Triplet<Integer, String, String>(split, ruleRight.leftChild, ruleRight.rightChild));
							
					  	  }
						}
					 }	
								
							
							
				
					//System.out.println("Exited rules loop");
					
					
				}
				//System.out.println("Grammar found for " + begin + " "+ end);
				//Now handle unary rules
				added = true;
				while (added){
					  added = false;	
					  Counter<String> count = parseScores.getCounter(begin+" "+ end);
					  PriorityQueue<String> countAsPriorityQueue = count.asPriorityQueue();
					  while (countAsPriorityQueue.hasNext()) {
					        String entry = countAsPriorityQueue.next();
					    	List<UnaryRule> unaryRules =grammar.getUnaryRulesByChild(entry);
					    	for(UnaryRule rule : unaryRules){
					    		double prob = rule.getScore()* parseScores.getCount(begin + " " +(end), entry);
					    		if(prob > parseScores.getCount(begin+" "+(end), rule.parent)){
					    		    parseScores.setCount(begin+" "+(end), rule.parent,prob);
					    		    
					    		    backHash.put(begin + " "+ (end) + " "+rule.parent, new Triplet<Integer, String, String>(-1, entry , null));
					    			added = true; 
					    		 }
					    		
					    	  }
					      	
					      }					  
					}			
				
				//System.out.println("Unaries dealt for " + begin + " "+ end);
				  
			  }				
			}
			
			
			// Create and return the parse tree
			Tree<String> parseTree= new Tree<String>("null");
			//System.out.println(parseScores.getCounter(0+" "+sentence.size()).toString());
			String parent = parseScores.getCounter(0+" "+sentence.size()).argMax();
			if(parent == null){
				System.out.println(parseScores.getCounter(0+ " " + sentence.size()).toString());
				System.out.println("THIS IS WEIRD");
			}
			parent = "ROOT";
			parseTree=getParseTreeOld(sentence,backHash,0,sentence.size(),parent);
			//System.out.println("PARSE SCORES");
		//	System.out.println(parseScores.toString());
			//System.out.println("BACK HASH");
			//System.out.println(backHash.toString());
		//	parseTree = addRoot(parseTree);
			//System.out.println(parseTree.toString());
			//return parseTree;
			return TreeAnnotations.unAnnotateTree(parseTree);
		}
		
		public Tree<String> getBestParse(List<String> sentence) {
			// This implements the CKY algorithm
			int nEntries = sentence.size();

			// hashmap to store back rules
			HashMap<Triplet<Integer, Integer, String>, Triplet<Integer,String,String>> backHash =
					new HashMap<Triplet<Integer, Integer, String>, Triplet<Integer,String,String>>();

			// more efficient access with arrays, but must cast each time :(
			@SuppressWarnings("unchecked")
			Counter<String>[][] parseScores = (Counter<String>[][])(new Counter[nEntries][nEntries]);
			
			for (int i = 0; i < nEntries; i++) {
				for (int j = 0; j < nEntries; j++) {
					parseScores[i][j] = new Counter<String>();
				}
			}
						
			System.out.println(sentence.toString());
			// First deal with the lexicons 
			int index =0;
			int span =1;// All spans are 1 at the lexicon level
			for (String word : sentence) {
				for (String tag : lexicon.getAllTags()) {
					double score = lexicon.scoreTagging(word, tag);
					if(score >=0.0){ // This lexicon may generate this word
					  //We use a counter map in order to store the scores for this sentence parse.
					  parseScores[index][index+span -1].setCount(tag, score);
					  	
					}
					
				}
				index = index +1;
				
			}
			
			// handle unary rules now 
			
			//System.out.println("Lexicons found");
			boolean added = true;
			
			while (added){
				added = false;	
				for( index =0; index < sentence.size(); index++){
				  // For each index+ span pair, get the counter.
				  Counter<String> count = parseScores[index][index+span -1];
				  PriorityQueue<String > countAsPQ = count.asPriorityQueue();
			      while( countAsPQ.hasNext()) {
			    	 String entry = countAsPQ.next();
			      //System.out.println("I am fine here!!");
			    	 List<UnaryRule> unaryRules =grammar.getUnaryRulesByChild(entry);
			    	 for(UnaryRule rule : unaryRules){
			    		 //These are the unary rules which might give rise to the above preterminal
			    		 double prob = rule.getScore()*parseScores[index][index+span - 1].getCount(entry);
			    	     if(prob > parseScores[index][index+span - 1].getCount(rule.parent)) {
			    		     parseScores[index][index+span - 1].setCount(rule.parent,prob);
			    		     backHash.put(new Triplet<Integer, Integer, String>(index, index+span, rule.parent), new Triplet<Integer, String, String>(-1, entry , null));
			    		     added = true; 
			    		 }
			    	 }
			      }
				}
			}
			//System.out.println("Lexicon unaries dealt with");
						
			// Now work with the grammar to produce higher level probabilities 
			for(span =2 ; span <= sentence.size(); span++ ){
			  for(int begin =0 ; begin <= (sentence.size()-span); begin++){
				int end = begin + span;  
				for(int split = begin +1 ; split <= end -1;split++){
					Counter<String> countLeft = parseScores[begin][split - 1];
					Counter<String> countRight = parseScores[split][end - 1];
					//List<BinaryRule> leftRules= new ArrayList<BinaryRule>();
					HashMap<Integer, BinaryRule> leftMap = new HashMap<Integer, BinaryRule>();
					//List<BinaryRule> rightRules=new ArrayList<BinaryRule>();
					HashMap<Integer, BinaryRule> rightMap = new HashMap<Integer, BinaryRule>();

					for (String entry: countLeft.keySet()){
						for(BinaryRule rule: grammar.getBinaryRulesByLeftChild(entry)){
							if(!leftMap.containsKey(rule.hashCode())){
								leftMap.put(rule.hashCode(), rule);
							}
						}
					}
					
					for(String entry: countRight.keySet()){
						for(BinaryRule rule: grammar.getBinaryRulesByRightChild(entry)){
							if(!rightMap.containsKey(rule.hashCode())){
								rightMap.put(rule.hashCode(), rule);
							}
						}

					}
					
					//System.out.println("About to enter the rules loops");
					for(Integer ruleHash: leftMap.keySet()){	
						if(rightMap.containsKey(ruleHash)){
							BinaryRule ruleRight = rightMap.get(ruleHash);
						  double prob = ruleRight.getScore()*parseScores[begin][split - 1].getCount(ruleRight.leftChild) *parseScores[split][end - 1].getCount(ruleRight.rightChild);
						  //System.out.println(begin+" "+ end +" "+ ruleRight.parent+ " "+ prob);
						  if(prob > parseScores[begin][end - 1].getCount(ruleRight.parent)){
							  //System.out.println(begin+" "+ end +" "+ ruleRight.parent+ " "+ prob);
							 //System.out.println("parentrule :"+ ruleRight.getParent());
						      parseScores[begin][end - 1].setCount(ruleRight.getParent(),prob);
							  backHash.put(new Triplet<Integer, Integer, String>(begin, end, ruleRight.parent), new Triplet<Integer, String, String>(split, ruleRight.leftChild, ruleRight.rightChild));
							
					  	  }
						}
					 }	
								
							
							
				
					//System.out.println("Exited rules loop");
					
					
				}
				//System.out.println("Grammar found for " + begin + " "+ end);
				//Now handle unary rules
				added = true;
				while (added){
					  added = false;	
					  Counter<String> count = parseScores[begin][end - 1];
					  PriorityQueue<String> countAsPriorityQueue = count.asPriorityQueue();
					  while (countAsPriorityQueue.hasNext()) {
					        String entry = countAsPriorityQueue.next();
					    	List<UnaryRule> unaryRules =grammar.getUnaryRulesByChild(entry);
					    	for(UnaryRule rule : unaryRules){
					    		double prob = rule.getScore()* parseScores[begin][end - 1].getCount(entry);
					    		if(prob > parseScores[begin][end - 1].getCount(rule.parent)){
					    		    parseScores[begin][end - 1].setCount(rule.parent,prob);
					    		    
					    		    backHash.put(new Triplet<Integer, Integer, String>(begin, end, rule.parent), new Triplet<Integer, String, String>(-1, entry , null));
					    			added = true; 
					    		 }
					    		
					    	  }
					      	
					      }					  
					}			
				
				//System.out.println("Unaries dealt for " + begin + " "+ end);
				  
			  }				
			}
			
			
			// Create and return the parse tree
			Tree<String> parseTree= new Tree<String>("null");
			//System.out.println(parseScores.getCounter(0+" "+sentence.size()).toString());
			
			// Pick the argmax
			String parent = parseScores[0][nEntries -1].argMax();
			
			// Or pick root. This second one is preferred since sentences are meant to have ROOT as their root node.
			parent = "ROOT";
			parseTree=getParseTree(sentence,backHash,0,sentence.size(),parent);
			//System.out.println("PARSE SCORES");
		//	System.out.println(parseScores.toString());
			//System.out.println("BACK HASH");
			//System.out.println(backHash.toString());
		//	parseTree = addRoot(parseTree);
			//System.out.println(parseTree.toString());
			//return parseTree;
			return TreeAnnotations.unAnnotateTree(parseTree);
		}
		
		private Tree<String> addRoot(Tree<String> tree) {
			return new Tree<String>("ROOT", Collections.singletonList(tree));
		}
		
		public Tree<String> getParseTree(List<String> sentence, HashMap<Triplet<Integer, Integer, String>, Triplet<Integer, String, String>> backHash,int begin, int end,String parent){
			// Start from the root and keep going down till you reach the leafs.
			if(begin == end -1){
				Triplet<Integer, String, String> triplet = backHash.get(new Triplet<Integer, Integer, String>(begin, end, parent));
				int split=-1;
				if(triplet != null){
					split = triplet.getFirst();	
				}

				Tree<String> topTree = new Tree<String>(parent);
				Tree<String> tree = topTree;
				while(triplet!=null && split == -1){
					
					Tree<String> singleTree = new Tree<String>(triplet.getSecond());
					tree.setChildren( Collections.singletonList(singleTree));

					triplet = backHash.get(new Triplet<Integer, Integer, String>(begin, end, triplet.getSecond()));
					if(triplet!=null){
						split = triplet.getFirst();
					}
					tree = tree.getChildren().get(0);

				}
				
				tree.setChildren(Collections.singletonList(new Tree<String>(sentence.get(begin))));
				return topTree;
			}

			Triplet<Integer, String, String> triplet = backHash.get(new Triplet<Integer, Integer, String>(begin, end, parent));

			if(triplet==null){
				System.out.println(begin+ " " + end+ " "+parent);
			}
			int split = triplet.getFirst();
			Tree<String> topTree = new Tree<String>(parent);
			Tree<String> tree = topTree;
			
	     	while(split == -1){
				Tree<String> singleTree = new Tree<String>(triplet.getSecond());
				tree.setChildren( Collections.singletonList(singleTree));
				triplet = backHash.get(new Triplet<Integer, Integer, String>(begin, end, triplet.getSecond()));
				if(triplet!=null){
				  split = triplet.getFirst();
				}
				tree = tree.getChildren().get(0);

			}

			Tree<String> leftTree = getParseTree(sentence,backHash,begin,split,triplet.getSecond());
			Tree<String> rightTree = getParseTree(sentence,backHash,split,end,triplet.getThird());

			List<Tree<String>> children = new ArrayList<Tree<String>>();
			children.add(leftTree);
			children.add(rightTree);
			tree.setChildren(children);
			return topTree;

		}

		public Tree<String> getParseTreeOld(List<String> sentence,HashMap<String, Triplet<Integer, String, String>> backHash,int begin, int end,String parent){
			// Start from the root and keep going down till you reach the leafs.
			//System.out.println("In recursion!!");
			if(begin == end -1){

				if((begin +" " + end).equals("0 1")){
					//System.out.println("CounterMap");
					//System.out.println(parseScores.getCounter(begin+" "+end).toString());
					//backHash.get(begin+ " " + end+ " "+parent);
				}

				
				//String parent = parseScores.getCounter(begin+" "+end).argMax();

				//System.out.println("Terminal cond :"+begin+ " "+ end+ " "+parent);
				Triplet<Integer, String, String> triplet = backHash.get(begin+ " " + end+ " "+parent);
				int split=-1;
				if(triplet != null){
					split = triplet.getFirst();
				
				}
				if((begin +" " + end).equals("0 1")){
					//System.out.println("CounterMap");
					//System.out.println(parseScores.getCounter(begin+" "+end).toString());
					//System.out.println(backHash.get(begin+ " " + end+ " "+parent).toString());
				}

				Tree<String> topTree = new Tree<String>(parent);
				Tree<String> tree = topTree;
				while(triplet!=null && split == -1){
					
					Tree<String> singleTree = new Tree<String>(triplet.getSecond());
					tree.setChildren( Collections.singletonList(singleTree));

					triplet = backHash.get(begin+ " " + end+ " "+triplet.getSecond());
					if(triplet!=null){
						split = triplet.getFirst();
					}
					tree = tree.getChildren().get(0);

				}
				
				//return new Tree<String>(tree.getLabel(), ));
				tree.setChildren(Collections.singletonList(new Tree<String>(sentence.get(begin))));
				return topTree;
			}
			
			/*if((begin +" " + end).equals("1 5")){
				System.out.println("CounterMap");
				System.out.println(parseScores.getCounter(begin+" "+end).toString());
				//backHash.get(begin+ " " + end+ " "+parent);
			}*/
			//String parent = parseScores.getCounter(begin+" "+end).argMax();
			//System.out.println(parent);
			Triplet<Integer, String, String> triplet = backHash.get(begin+ " " + end+ " "+parent);
			//System.out.println(triplet.getSecond() + "  " + triplet.getFirst());

			if((begin +" " + end).equals("0 6")){
				//System.out.println("CounterMap");
				//System.out.println(parent);
				//System.out.println(backHash.get(begin+ " " + end+ " "+parent).toString());
			}

			if(triplet==null){
				System.out.println(begin+ " " + end+ " "+parent);
			}
			int split = triplet.getFirst();
			Tree<String> topTree = new Tree<String>(parent);
			Tree<String> tree = topTree;
			//System.out.println("parent : " +parent);
	     	while(split == -1){
				//System.out.println(tree.toString());
				Tree<String> singleTree = new Tree<String>(triplet.getSecond());
				//System.out.println(triplet.getSecond());
				tree.setChildren( Collections.singletonList(singleTree));
				//System.out.println(tree.toString());
				//System.out.println("XXXXXXXXXXXXXXXXXXXXXXXXXXXxx");
				//System.out.println(triplet.getSecond());
				triplet = backHash.get(begin+ " " + end+ " "+triplet.getSecond());
				if(triplet!=null){
				  split = triplet.getFirst();
				}
				tree = tree.getChildren().get(0);

			}
			//System.out.println(tree.toString());

			Tree<String> leftTree = getParseTreeOld(sentence,backHash,begin,split,triplet.getSecond());
			Tree<String> rightTree = getParseTreeOld(sentence,backHash,split,end,triplet.getThird());
			//System.out.println("leftTree: "+ leftTree.toString());
			//System.out.println("rightTree :" +rightTree.toString());
			//System.out.println("topTree :"+topTree.toString());
			List<Tree<String>> children = new ArrayList<Tree<String>>();
			children.add(leftTree);
			children.add(rightTree);
			tree.setChildren( children);
			return topTree;

		}
		
	}


	// BaselineParser =============================================================

	/**
	 * Baseline parser (though not a baseline I've ever seen before).  Tags the
	 * sentence using the baseline tagging method, then either retrieves a known
	 * parse of that tag sequence, or builds a right-branching parse for unknown
	 * tag sequences.
	 */
	public static class BaselineParser implements Parser {

		CounterMap<List<String>,Tree<String>> knownParses;
		CounterMap<Integer,String> spanToCategories;
		Lexicon lexicon;

		public void train(List<Tree<String>> trainTrees) {
			lexicon = new Lexicon(trainTrees);
			knownParses = new CounterMap<List<String>, Tree<String>>();
			spanToCategories = new CounterMap<Integer, String>();
			for (Tree<String> trainTree : trainTrees) {
				List<String> tags = trainTree.getPreTerminalYield();
				knownParses.incrementCount(tags, trainTree, 1.0);
				tallySpans(trainTree, 0);
			}
		}

		public Tree<String> getBestParse(List<String> sentence) {
			List<String> tags = getBaselineTagging(sentence);
			if (knownParses.keySet().contains(tags)) {
				return getBestKnownParse(tags, sentence);
			}
			return buildRightBranchParse(sentence, tags);
		}

		/* Builds a tree that branches to the right.  For pre-terminals it
		 * uses the most common tag for the word in the training corpus.
		 * For all other non-terminals it uses the tag that is most common
		 * in training corpus of tree of the same size span as the tree
		 * that is being labeled. */
		private Tree<String> buildRightBranchParse(List<String> words, List<String> tags) {
			int currentPosition = words.size() - 1;
			Tree<String> rightBranchTree = buildTagTree(words, tags, currentPosition);
			while (currentPosition > 0) {
				currentPosition--;
				rightBranchTree = merge(buildTagTree(words, tags, currentPosition),
						rightBranchTree);
			}
			rightBranchTree = addRoot(rightBranchTree);
			return rightBranchTree;
		}

		private Tree<String> merge(Tree<String> leftTree, Tree<String> rightTree) {
			int span = leftTree.getYield().size() + rightTree.getYield().size();
			String mostFrequentLabel = spanToCategories.getCounter(span).argMax();
			List<Tree<String>> children = new ArrayList<Tree<String>>();
			children.add(leftTree);
			children.add(rightTree);
			return new Tree<String>(mostFrequentLabel, children);
		}

		private Tree<String> addRoot(Tree<String> tree) {
			return new Tree<String>("ROOT", Collections.singletonList(tree));
		}

		private Tree<String> buildTagTree(List<String> words,
				List<String> tags,
				int currentPosition) {
			Tree<String> leafTree = new Tree<String>(words.get(currentPosition));
			Tree<String> tagTree = new Tree<String>(tags.get(currentPosition), 
					Collections.singletonList(leafTree));
			return tagTree;
		}

		private Tree<String> getBestKnownParse(List<String> tags, List<String> sentence) {
			Tree<String> parse = knownParses.getCounter(tags).argMax().deepCopy();
			parse.setWords(sentence);
			return parse;
		}

		private List<String> getBaselineTagging(List<String> sentence) {
			List<String> tags = new ArrayList<String>();
			for (String word : sentence) {
				String tag = getBestTag(word);
				tags.add(tag);
			}
			return tags;
		}

		private String getBestTag(String word) {
			double bestScore = Double.NEGATIVE_INFINITY;
			String bestTag = null;
			for (String tag : lexicon.getAllTags()) {
				double score = lexicon.scoreTagging(word, tag);
				if (bestTag == null || score > bestScore) {
					bestScore = score;
					bestTag = tag;
				}
			}
			return bestTag;
		}

		private int tallySpans(Tree<String> tree, int start) {
			if (tree.isLeaf() || tree.isPreTerminal()) 
				return 1;
			int end = start;
			for (Tree<String> child : tree.getChildren()) {
				int childSpan = tallySpans(child, end);
				end += childSpan;
			}
			String category = tree.getLabel();
			if (! category.equals("ROOT"))
				spanToCategories.incrementCount(end - start, category, 1.0);
			return end - start;
		}

	}


	// TreeAnnotations ============================================================

	/**
	 * Class which contains code for annotating and binarizing trees for
	 * the parser's use, and debinarizing and unannotating them for
	 * scoring.
	 */
	public static class TreeAnnotations {

		public static Tree<String> annotateTree(Tree<String> unAnnotatedTree) {

			// Currently, the only annotation done is a lossless binarization

			// TODO: change the annotation from a lossless binarization to a
			// finite-order markov process (try at least 1st and 2nd order)

			// TODO : mark nodes with the label of their parent nodes, giving a second
			// order vertical markov process
			

			System.out.println("Old Tree: " + unAnnotatedTree);
			Tree<String> secondOrderVerticalMarkovizationTree =
				markovizeTree(unAnnotatedTree, null);//unAnnotatedTree;
			
			System.out.println("2nd order Markov Tree: " + secondOrderVerticalMarkovizationTree);
			Tree<String> newTree = binarizeTree(secondOrderVerticalMarkovizationTree);
			System.out.println("New Tree: " + newTree);

			return newTree;

		}
		private static Tree<String> markovizeTree(Tree<String> tree, String parentLabel) {
			String label = tree.getLabel();
			
			// Tried using ^ but unannotate didn't remove it. Instead using - since that is properly removed
			if (parentLabel != null) {
				label = label + "-" + parentLabel;
			}
			
			// If you're a preterminal, don't bother markovizing
			if (tree.isPreTerminal())
				return tree;
			
			List<Tree<String>> children = new ArrayList<Tree<String>>();
			for (Tree<String> child : tree.getChildren()) {
				children.add(markovizeTree(child, tree.getLabel()));
			}
			
			Tree<String> newTree = new Tree<String>(label, children);
			
			return newTree;
		}

		private static Tree<String> binarizeTree(Tree<String> tree) {
			String label = tree.getLabel();
			if (tree.isLeaf())
				return new Tree<String>(label);
			if (tree.getChildren().size() == 1) {
				return new Tree<String>
				(label, 
						Collections.singletonList(binarizeTree(tree.getChildren().get(0))));
			}
			// I think it tries to binarize a binary tree. This is silly. Just binarize the subtrees.
			if (tree.getChildren().size() == 2) {

				List<Tree<String>> children = new ArrayList<Tree<String>>();
				children.add(binarizeTree(tree.getChildren().get(0)));
				children.add(binarizeTree(tree.getChildren().get(1)));
				return new Tree<String>(label, children);
			}
			
			// otherwise, it's a TERNARY-or-more local tree, 
			// so decompose it into a sequence of binary and unary trees.
			String intermediateLabel = "@"+label+"->";
			Tree<String> intermediateTree =
					binarizeTreeHelper(tree, 0, intermediateLabel);
			return new Tree<String>(label, intermediateTree.getChildren());
		}

		private static Tree<String> binarizeTreeHelper(Tree<String> tree,
				int numChildrenGenerated, 
				String intermediateLabel) {
			Tree<String> leftTree = tree.getChildren().get(numChildrenGenerated);
			List<Tree<String>> children = new ArrayList<Tree<String>>();
			children.add(binarizeTree(leftTree));
			
			// Wait! Don't binarize too much. The last child doesn't need to have a new node.
			// It can be paired with the 2nd to last child!
			if (numChildrenGenerated == tree.getChildren().size() - 2) {
				children.add(binarizeTree(tree.getChildren().get(numChildrenGenerated + 1)));
			} else if (numChildrenGenerated < tree.getChildren().size() - 1) {
				Tree<String> rightTree = 
						binarizeTreeHelper(tree, numChildrenGenerated + 1, 
								intermediateLabel + "_" + leftTree.getLabel());
				children.add(rightTree);
			}
			return new Tree<String>(intermediateLabel, children);
		} 

		public static Tree<String> unAnnotateTree(Tree<String> annotatedTree) {

			// Remove intermediate nodes (labels beginning with "@"
			// Remove all material on node labels which follow their base symbol 
			// (cuts at the leftmost -, ^, or : character)
			// Examples: a node with label @NP->DT_JJ will be spliced out, 
			// and a node with label NP^S will be reduced to NP

			Tree<String> debinarizedTree =
					Trees.spliceNodes(annotatedTree, new Filter<String>() {
						public boolean accept(String s) {
							return s.startsWith("@");
						}
					});
			Tree<String> unAnnotatedTree = 
					(new Trees.FunctionNodeStripper()).transformTree(debinarizedTree);
			return unAnnotatedTree;
		}
	}


	// Lexicon ====================================================================

	/**
	 * Simple default implementation of a lexicon, which scores word,
	 * tag pairs with a smoothed estimate of P(tag|word)/P(tag).
	 */
	public static class Lexicon {

		CounterMap<String,String> wordToTagCounters = new CounterMap<String, String>();
		double totalTokens = 0.0;
		double totalWordTypes = 0.0;
		Counter<String> tagCounter = new Counter<String>();
		Counter<String> wordCounter = new Counter<String>();
		Counter<String> typeTagCounter = new Counter<String>();

		public Set<String> getAllTags() {
			return tagCounter.keySet();
		}

		public boolean isKnown(String word) {
			return wordCounter.keySet().contains(word);
		}

		/* Returns a smoothed estimate of P(word|tag) */
		public double scoreTagging(String word, String tag) {
			double p_tag = tagCounter.getCount(tag) / totalTokens;
			double c_word = wordCounter.getCount(word);
			double c_tag_and_word = wordToTagCounters.getCount(word, tag);
			if (c_word < 10) { // rare or unknown
				c_word += 1.0;
				c_tag_and_word += typeTagCounter.getCount(tag) / totalWordTypes;
			}
			double p_word = (1.0 + c_word) / (totalTokens + totalWordTypes);
			double p_tag_given_word = c_tag_and_word / c_word;
			return p_tag_given_word / p_tag * p_word;
		}

		/* Builds a lexicon from the observed tags in a list of training trees. */
		public Lexicon(List<Tree<String>> trainTrees) {
			for (Tree<String> trainTree : trainTrees) {
				List<String> words = trainTree.getYield();
				List<String> tags = trainTree.getPreTerminalYield();
				for (int position = 0; position < words.size(); position++) {
					String word = words.get(position);
					String tag = tags.get(position);
					tallyTagging(word, tag);
				}
			}
		}

		private void tallyTagging(String word, String tag) {
			if (! isKnown(word)) {
				totalWordTypes += 1.0;
				typeTagCounter.incrementCount(tag, 1.0);
			}
			totalTokens += 1.0;
			tagCounter.incrementCount(tag, 1.0);
			wordCounter.incrementCount(word, 1.0);
			wordToTagCounters.incrementCount(word, tag, 1.0);
		}
	}


	// Grammar ====================================================================

	/**
	 * Simple implementation of a PCFG grammar, offering the ability to
	 * look up rules by their child symbols.  Rule probability estimates
	 * are just relative frequency estimates off of training trees.
	 */
	public static class Grammar {

		Map<String, List<BinaryRule>> binaryRulesByLeftChild = 
				new HashMap<String, List<BinaryRule>>();
		Map<String, List<BinaryRule>> binaryRulesByRightChild = 
				new HashMap<String, List<BinaryRule>>();
		Map<String, List<UnaryRule>> unaryRulesByChild = 
				new HashMap<String, List<UnaryRule>>();

		/* Rules in grammar are indexed by child for easy access when
		 * doing bottom up parsing. */
		public List<BinaryRule> getBinaryRulesByLeftChild(String leftChild) {
			return CollectionUtils.getValueList(binaryRulesByLeftChild, leftChild);
		}

		public List<BinaryRule> getBinaryRulesByRightChild(String rightChild) {
			return CollectionUtils.getValueList(binaryRulesByRightChild, rightChild);
		}

		public List<UnaryRule> getUnaryRulesByChild(String child) {
			return CollectionUtils.getValueList(unaryRulesByChild, child);
		}

		public String toString() {
			StringBuilder sb = new StringBuilder();
			List<String> ruleStrings = new ArrayList<String>();
			for (String leftChild : binaryRulesByLeftChild.keySet()) {
				for (BinaryRule binaryRule : getBinaryRulesByLeftChild(leftChild)) {
					ruleStrings.add(binaryRule.toString());
				}
			}
			for (String child : unaryRulesByChild.keySet()) {
				for (UnaryRule unaryRule : getUnaryRulesByChild(child)) {
					ruleStrings.add(unaryRule.toString());
				}
			}
			for (String ruleString : CollectionUtils.sort(ruleStrings)) {
				sb.append(ruleString);
				sb.append("\n");
			}
			return sb.toString();
		}

		private void addBinary(BinaryRule binaryRule) {
			CollectionUtils.addToValueList(binaryRulesByLeftChild, 
					binaryRule.getLeftChild(), binaryRule);
			CollectionUtils.addToValueList(binaryRulesByRightChild, 
					binaryRule.getRightChild(), binaryRule);
		}

		private void addUnary(UnaryRule unaryRule) {
			CollectionUtils.addToValueList(unaryRulesByChild, 
					unaryRule.getChild(), unaryRule);
		}

		/* A builds PCFG using the observed counts of binary and unary
		 * productions in the training trees to estimate the probabilities
		 * for those rules.  */ 
		public Grammar(List<Tree<String>> trainTrees) {
			Counter<UnaryRule> unaryRuleCounter = new Counter<UnaryRule>();
			Counter<BinaryRule> binaryRuleCounter = new Counter<BinaryRule>();
			Counter<String> symbolCounter = new Counter<String>();
			for (Tree<String> trainTree : trainTrees) {
				tallyTree(trainTree, symbolCounter, unaryRuleCounter, binaryRuleCounter);
			}
			for (UnaryRule unaryRule : unaryRuleCounter.keySet()) {
				double unaryProbability = 
						unaryRuleCounter.getCount(unaryRule) / 
						symbolCounter.getCount(unaryRule.getParent());
				unaryRule.setScore(unaryProbability);
				addUnary(unaryRule);
			}
			for (BinaryRule binaryRule : binaryRuleCounter.keySet()) {
				double binaryProbability = 
						binaryRuleCounter.getCount(binaryRule) / 
						symbolCounter.getCount(binaryRule.getParent());
				binaryRule.setScore(binaryProbability);
				addBinary(binaryRule);
			}
		}

		private void tallyTree(Tree<String> tree, Counter<String> symbolCounter,
				Counter<UnaryRule> unaryRuleCounter, 
				Counter<BinaryRule> binaryRuleCounter) {
			if (tree.isLeaf()) return;
			if (tree.isPreTerminal()) return;
			if (tree.getChildren().size() == 1) {
				UnaryRule unaryRule = makeUnaryRule(tree);
				symbolCounter.incrementCount(tree.getLabel(), 1.0);
				unaryRuleCounter.incrementCount(unaryRule, 1.0);
			}
			if (tree.getChildren().size() == 2) {
				BinaryRule binaryRule = makeBinaryRule(tree);
				symbolCounter.incrementCount(tree.getLabel(), 1.0);
				binaryRuleCounter.incrementCount(binaryRule, 1.0);
			}
			if (tree.getChildren().size() < 1 || tree.getChildren().size() > 2) {
				throw new RuntimeException("Attempted to construct a Grammar with an illegal tree: "+tree);
			}
			for (Tree<String> child : tree.getChildren()) {
				tallyTree(child, symbolCounter, unaryRuleCounter,  binaryRuleCounter);
			}
		}

		private UnaryRule makeUnaryRule(Tree<String> tree) {
			return new UnaryRule(tree.getLabel(), tree.getChildren().get(0).getLabel());
		}

		private BinaryRule makeBinaryRule(Tree<String> tree) {
			return new BinaryRule(tree.getLabel(), tree.getChildren().get(0).getLabel(), 
					tree.getChildren().get(1).getLabel());
		}
	}


	// BinaryRule =================================================================

	/* A binary grammar rule with score representing its probability. */
	public static class BinaryRule {

		String parent;
		String leftChild;
		String rightChild;
		double score;

		public String getParent() {
			return parent;
		}

		public String getLeftChild() {
			return leftChild;
		}

		public String getRightChild() {
			return rightChild;
		}

		public double getScore() {
			return score;
		}

		public void setScore(double score) {
			this.score = score;
		}

		public boolean equals(Object o) {
			if (this == o) return true;
			if (!(o instanceof BinaryRule)) return false;

			final BinaryRule binaryRule = (BinaryRule) o;

			if (leftChild != null ? !leftChild.equals(binaryRule.leftChild) : binaryRule.leftChild != null) 
				return false;
			if (parent != null ? !parent.equals(binaryRule.parent) : binaryRule.parent != null) 
				return false;
			if (rightChild != null ? !rightChild.equals(binaryRule.rightChild) : binaryRule.rightChild != null) 
				return false;

			return true;
		}

		public int hashCode() {
			int result;
			result = (parent != null ? parent.hashCode() : 0);
			result = 29 * result + (leftChild != null ? leftChild.hashCode() : 0);
			result = 29 * result + (rightChild != null ? rightChild.hashCode() : 0);
			return result;
		}

		public String toString() {
			return parent + " -> " + leftChild + " " + rightChild + " %% "+score;
		}

		public BinaryRule(String parent, String leftChild, String rightChild) {
			this.parent = parent;
			this.leftChild = leftChild;
			this.rightChild = rightChild;
		}
	}


	// UnaryRule ==================================================================

	/** A unary grammar rule with score representing its probability. */
	public static class UnaryRule {

		String parent;
		String child;
		double score;

		public String getParent() {
			return parent;
		}

		public String getChild() {
			return child;
		}

		public double getScore() {
			return score;
		}

		public void setScore(double score) {
			this.score = score;
		}

		public boolean equals(Object o) {
			if (this == o) return true;
			if (!(o instanceof UnaryRule)) return false;

			final UnaryRule unaryRule = (UnaryRule) o;

			if (child != null ? !child.equals(unaryRule.child) : unaryRule.child != null) return false;
			if (parent != null ? !parent.equals(unaryRule.parent) : unaryRule.parent != null) return false;

			return true;
		}

		public int hashCode() {
			int result;
			result = (parent != null ? parent.hashCode() : 0);
			result = 29 * result + (child != null ? child.hashCode() : 0);
			return result;
		}

		public String toString() {
			return parent + " -> " + child + " %% "+score;
		}

		public UnaryRule(String parent, String child) {
			this.parent = parent;
			this.child = child;
		}
	}


	// PCFGParserTester ===========================================================

	// Longest sentence length that will be tested on.
	private static int MAX_LENGTH = 20;

	private static void testParser(Parser parser, List<Tree<String>> testTrees) {
		EnglishPennTreebankParseEvaluator.LabeledConstituentEval<String> eval = 
				new EnglishPennTreebankParseEvaluator.LabeledConstituentEval<String>
		(Collections.singleton("ROOT"), 
				new HashSet<String>(Arrays.asList(new String[] {"''", "``", ".", ":", ","})));
		for (Tree<String> testTree : testTrees) {
			List<String> testSentence = testTree.getYield();
			if (testSentence.size() > MAX_LENGTH)
				continue;
			Tree<String> guessedTree = parser.getBestParse(testSentence);
			System.out.println("Guess:\n"+Trees.PennTreeRenderer.render(guessedTree));
			System.out.println("Gold:\n"+Trees.PennTreeRenderer.render(testTree));
			eval.evaluate(guessedTree, testTree);
		}
		eval.display(true);
	}

	private static List<Tree<String>> readTrees(String basePath, int low,
			int high) {
		Collection<Tree<String>> trees = PennTreebankReader.readTrees(basePath,
				low, high);
		// normalize trees
		Trees.TreeTransformer<String> treeTransformer = new Trees.StandardTreeNormalizer();
		List<Tree<String>> normalizedTreeList = new ArrayList<Tree<String>>();
		for (Tree<String> tree : trees) {
			Tree<String> normalizedTree = treeTransformer.transformTree(tree);
			// System.out.println(Trees.PennTreeRenderer.render(normalizedTree));
			normalizedTreeList.add(normalizedTree);
		}
		return normalizedTreeList;
	}

	public static void main(String[] args) {

		// set up default options ..............................................
		Map<String, String> options = new HashMap<String, String>();
		options.put("-path",      "/afs/ir/class/cs224n/pa2/data/");
		options.put("-data",      "miniTest");
		options.put("-parser",    "cs224n.assignments.PCFGParserTester$BaselineParser");
		options.put("-maxLength", "20");

		// let command-line options supersede defaults .........................
		options.putAll(CommandLineUtils.simpleCommandLineParser(args));
		System.out.println("PCFGParserTester options:");
		for (Map.Entry<String, String> entry: options.entrySet()) {
			System.out.printf("  %-12s: %s%n", entry.getKey(), entry.getValue());
		}
		System.out.println();

		MAX_LENGTH = Integer.parseInt(options.get("-maxLength"));

		Parser parser;
		try {
			Class parserClass = Class.forName(options.get("-parser"));
			parser = (Parser) parserClass.newInstance();
		} catch (Exception e) {
			throw new RuntimeException(e);
		}
		System.out.println("Using parser: " + parser);

		String basePath = options.get("-path");
		String dataSet = options.get("-data");
		if (!basePath.endsWith("/"))
			basePath += "/";
		//basePath += dataSet;
		System.out.println("Data will be loaded from: " + basePath + "\n");

		List<Tree<String>> trainTrees = new ArrayList<Tree<String>>(),
				validationTrees = new ArrayList<Tree<String>>(),
				testTrees = new ArrayList<Tree<String>>();

		if (!basePath.endsWith("/"))
			basePath += "/";
		basePath += dataSet;
		if (dataSet.equals("miniTest")) {
			System.out.print("Loading training trees...");
			trainTrees = readTrees(basePath, 1, 3);
			System.out.println("done.");
			System.out.print("Loading test trees...");
			testTrees = readTrees(basePath, 4, 4);
			System.out.println("done.");
		}
		else if (dataSet.equals("treebank")) {
			System.out.print("Loading training trees...");
			trainTrees = readTrees(basePath, 200, 2199);
			System.out.println("done.");
			System.out.print("Loading validation trees...");
			validationTrees = readTrees(basePath, 2200, 2202);
			System.out.println("done.");
			System.out.print("Loading test trees...");
			testTrees = readTrees(basePath, 2300, 2319); //2301);
			System.out.println("done.");
		}
		else {
			throw new RuntimeException("Bad data set mode: "+ dataSet+", use miniTest, or treebank."); 
		}
		parser.train(trainTrees);
		testParser(parser, testTrees);
	}
}
