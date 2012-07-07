#!/usr/bin/python

import subprocess
import sys, os, shutil
import string,re

perf_test_pat = re.compile("CudaDMA Sequential Performance Test")
alignment_pat = re.compile("\s+ALIGNMENT - (?P<align>[0-9]+)")
elmt_size_pat = re.compile("\s+ELEMENT SIZE - (?P<size>[0-9]+)")
true_spec_pat = re.compile("\s+WARP SPECIALIZED - true")
false_spec_pat= re.compile("\s+WARP SPECIALIZED - false")
buffering_pat = re.compile("\s+BUFFERING - (?P<buff>\w+)")
dma_warps_pat = re.compile("\s+DMA WARPS - (?P<warps>[0-9]+)")
cta_per_sm_pat= re.compile("\s+CTAs/SM - (?P<persm>[0-9]+)")
loop_iter_pat = re.compile("\s+LOOP ITERATIONS - (?P<loops>[0-9]+)")
total_cta_pat = re.compile("\s+Total CTAS - (?P<ctas>[0-9]+)")
total_mem_pat = re.compile("\s+Total memory - (?P<mem>[0-9]+)")
total_perf_pat= re.compile("Performance - (?P<perf>[0-9\.]+)")


list_pat = re.compile("list (?P<modifier>\w+) (?P<count>[0-9]+) where(?P<constraints>(\s+\w+[><=]+\w+)*)")
total_pat= re.compile("total where(?P<constraints>(\s+\w+[><=]+\w+)*)")
aver_pat = re.compile("average where(?P<constraints>(\s+\w+[><=]+\w+)*)")
dev_pat  = re.compile("deviation where(?P<constraints>(\s+\w+[><=]+\w+)*)")
cons_pat = re.compile("(?P<left>\w+)(?P<op>[><=]+)(?P<right>\w+)")
pop_pat  = re.compile("pop")
clear_pat= re.compile("clear")
quit_pat = re.compile("quit")

class Experiment(object):
    def __init__(self):
        self.alignment = None 
	self.elmt_size = None 
	self.specialized = None
	self.buffering = None
	self.total_warps = 0
	self.ctas_per_sm = None
	self.loop_iters = None
	self.total_ctas = None
	self.total_mem = None
	self.perf = None

    def complete(self):
        if self.alignment <> None and self.elmt_size <> None and \
	   self.specialized <> None and self.ctas_per_sm <> None and \
	   self.loop_iters <> None and self.total_ctas <> None and \
	   self.total_mem <> None and self.perf <> None:
	    if self.specialized:
	        if self.buffering <> None:
		    return True
		return False
	    return True
	return False

    def meets(self,constraints):
        for c in constraints:
	    if not c.evaluate(self):
	        return False
        return True

    def print_experiment(self,indent):
        print indent+"Experiment:"
	print indent+"  Alignment - "+str(self.alignment)
	print indent+"  Element Size - "+str(self.elmt_size)
	print indent+"  Specialized - "+str(self.specialized)
	if self.specialized:
	    print indent+"  Buffering - "+str(self.buffering) 
	print indent+"  Total Warps - "+str(self.total_warps)
	print indent+"  CTAs/SM - "+str(self.ctas_per_sm)
	print indent+"  Loop Iterations - "+str(self.loop_iters)
	print indent+"  Total CTAs - "+str(self.total_ctas)
	print indent+"  Total Memory - "+str(self.total_mem)
	print indent+"  Performance - "+str(self.perf)+" GB/s"
	#print ""

class Constraint(object):
    def __init__(self,field,op,value):
        self.field = None
	self.op = None
	self.value = None

	# Try converting to an integer first
	try:
	    self.value = int(value)
	except ValueError:
	    try:
	        self.value = float(value)
            except ValueError:
	        if value == "true" or value == "True":
		    self.value = True
		elif value == "false" or value == "False":
		    self.value = False
		else:
	            self.value = value 
	self.field = self.parse_field_name(field)
	self.op = self.parse_op_name(op)

    def is_valid(self):
        if self.field == None or self.op == None or self.value == None:
	    return False
        return True

    def parse_field_name(self,name):
        # These correspond to the field names of the Experiments
	if name == "alignment":
	    return name
	if name == "size" or name == "elmt_size":
	    return "elmt_size"
        if name == "warps":
	    return "total_warps"
	if name == "buffering":
	    return name
	if name == "specialized":
	    return name
	if name == "CTAperSM":
	    return "ctas_per_sm"
	if name == "loops" or name == "iters":
	    return "loop_iters"
	if name == "CTAs":
	    return "total_ctas"
	if name == "Mem" or name == "Memory":
	    return "total_mem"
	if name == "Perf":
	    return "perf"
	return None

    def parse_op_name(self,op):
        if op == "<":
	    return lambda x,y: x < y
	if op == "<=":
	    return lambda x,y: x <= y
	if op == "=" or op == "==":
	    return lambda x,y: x==y
	if op == ">":
	    return lambda x,y: x > y
	if op == ">=":
	    return lambda x,y: x >= y
	return None

    def evaluate(self,experiment):
        field_val = getattr(experiment,self.field)
        return self.op(field_val,self.value)

def find_best(result_set):
    best_ex = None
    best_perf = 0
    for ex in result_set:
        if ex.perf > best_perf:
	    best_ex = ex
	    best_perf = ex.perf
    assert best_ex <> None
    print "Best performance:"
    best_ex.print_experiment()

def parse_results(result_set,file_name):
    cur_obj = None
    f = open(file_name,'r')
    for line in f:
        m = perf_test_pat.match(line)
	if m <> None:
	    if cur_obj <> None:
	        #assert cur_obj.complete()
		if not cur_obj.complete():
		    cur_obj.print_experiment()
		    assert False
	        result_set.add(cur_obj)
	    cur_obj = Experiment()
	    continue
	m = alignment_pat.match(line)
	if m <> None:
	    cur_obj.alignment = int(m.group('align'))
	    continue
	m = elmt_size_pat.match(line)
	if m <> None:
	    cur_obj.elmt_size = int(m.group('size'))
	    continue
	m = true_spec_pat.match(line)
	if m <> None:
	    cur_obj.specialized = True
	    continue
	m = false_spec_pat.match(line)
	if m <> None:
	    cur_obj.specialized = False
	    continue
	m = buffering_pat.match(line)
	if m <> None:
	    cur_obj.buffering = m.group('buff')
	    continue
	m = dma_warps_pat.match(line)
	if m <> None:
	    cur_obj.total_warps = int(m.group('warps'))
	    continue
	m = cta_per_sm_pat.match(line)
	if m <> None:
	    cur_obj.ctas_per_sm = int(m.group('persm'))
	    continue
	m = loop_iter_pat.match(line)
	if m <> None:
	    cur_obj.loop_iters = int(m.group('loops'))
	    continue
	m = total_cta_pat.match(line)
	if m <> None:
	    cur_obj.total_ctas = int(m.group('ctas'))
	    continue
	m = total_mem_pat.match(line)
	if m <> None:
	    cur_obj.total_mem = int(m.group('mem'))
	    continue
	m = total_perf_pat.match(line)
	if m <> None:
	    cur_obj.perf = float(m.group('perf'))
	    continue
    if cur_obj <> None and cur_obj.complete():
        result_set.add(cur_obj)
    f.close()

def parse_constraints(constraint_strings):
    #print "Parsing constraints "+str(constraint_strings)
    result = set()
    for c in str.split(constraint_strings):
        m = cons_pat.match(c)
	assert m <> None
	constraint = Constraint(m.group('left'),m.group('op'),m.group('right'))
	if constraint.is_valid():
	    result.add(constraint)
	else:
	    print "Invalid constraint: "+str(c)
    #print "Total constraints "+str(len(result))
    return result

def find_experiments(experiments,constraints):
    result = set()
    for ex in experiments:
        if ex.meets(constraints):
	    result.add(ex)
    return result

def execute_list_command(command, count, constraint_strings, experiments):
    constraints = parse_constraints(constraint_strings) 
    if constraints == None:
        return False
    matches = find_experiments(experiments,constraints)
    if command == "top":
        total_printed = 0
        for ex in sorted(matches,key=lambda x: x.perf,reverse=True):
	    total_printed = total_printed + 1
	    print "Experiment "+str(total_printed)
	    ex.print_experiment('  ')
	    print ""
	    if total_printed == count:
	        break
        if total_printed < count:
	    print "Total results "+str(len(matches))+" less than "+str(count)
	    return False
	return True
    else:
        print "Unimplemented command "+str(command)
	return False

def execute_total_command(constraint_strings,experiments):
    constraints = parse_constraints(constraint_strings)
    if constraints == None:
        return False
    matches = find_experiments(experiments,constraints)
    print "Total: "+str(len(matches))+" matching experiments"
    return True

def execute_average_command(constraint_strings,experiments):
    constraints = parse_constraints(constraint_strings)
    if constraints == None:
        return False
    matches = find_experiments(experiments,constraints)
    if len(matches) == 0:
        print "There were no results that satisfied the constraints"
	return False
    total_perf = 0.0
    for ex in matches:
        total_perf = total_perf + ex.perf
    avg_perf = total_perf/len(matches)
    print "Average perf for "+str(len(matches))+" results was "+str(avg_perf)+" GB/s"
    return True

def execute_deviation_command(constraint_strings,experiments):
    constraints = parse_constraints(constraint_strings)
    if constraints == None:
        return False
    print "Not implemented"
    return False 

if __name__ == "__main__":
    assert len(sys.argv) > 1
    results = set()
    parse_results(results,sys.argv[1])
    prev_cmd = list()
    while True:
        cmd = "" 
	for pc in prev_cmd:
	    cmd = cmd + pc + ' '
        temp_cmd = raw_input('Enter a command >: '+cmd) 
        cmd = cmd + temp_cmd	
	# These have to go first
        m = pop_pat.match(temp_cmd)
	if m <> None:
	    prev_cmd.pop()
	    continue
	m = clear_pat.match(temp_cmd)
	if m <> None:
	    prev_cmd = list() 
	    continue
	m = quit_pat.match(temp_cmd)
	if m <> None:
	    break
	m = list_pat.match(cmd)
	if m <> None:
	    if execute_list_command(m.group('modifier'),int(m.group('count')),m.group('constraints'),results):
	        prev_cmd.append(temp_cmd)
	    continue
	m = total_pat.match(cmd)
	if m <> None:
	    if execute_total_command(m.group('constraints'),results):
	        prev_cmd.append(temp_cmd)
	    continue
	m = aver_pat.match(cmd)
	if m <> None:
	    if execute_average_command(m.group('constraints'),results):
	        prev_cmd.append(temp_cmd)
	    continue
        m = dev_pat.match(cmd)
	if m <> None:
	    if execute_deviation_command(m.group('constraints'),results):
	        prev_cmd.append(temp_cmd)
	    continue
	
	print "Illegal command! "+str(cmd)+"  Type 'quit' to quit"

