function code(s:ansistring;m:boolean):ansistring;
var x:longword;mode:integer;st:ansistring;
begin
if m then st:='ccc' else st:='';
if m then mode:=1 else mode:=-1;
for x:=1 to length(s) do st:=st+chr(ord(s[x])+mode);
code:=st;
end;
function pass(st:ansistring):ansistring;
var out:ansistring;
begin
if copy(st,1,3)='ccc' then
out:=code(copy(st,4,length(ST)-3),false)
else out:=code(st,true);
pass:=out;
end;